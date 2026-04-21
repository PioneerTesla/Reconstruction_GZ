"""
Semantic Coding PRI Pattern Reconstruction
============================================
Based on: Yuan et al., "Reconstruction of Radar Pulse Repetition Pattern
via Semantic Coding of Intercepted Pulse Trains", IEEE TAES, 2023.

Core idea — Minimum Description Length (MDL):
  1. Treat the PRI sequence as a "document" composed of repeating "words"
  2. Build a dictionary of PRI sub-patterns via iterative BPE-style merging
  3. Accept merges only when total description length decreases
  4. The dominant dictionary word = the radar's PRI pattern
  5. Reconstruct the clean sequence by tiling the found pattern
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import List, Optional, Tuple


class SemanticCodingReconstructor:
    """Model-driven (non-learning) PRI pattern reconstruction."""

    def __init__(
        self,
        cluster_tol: float = 8.0,
        max_period: int = 40,
        max_merge_iter: int = 200,
    ):
        self.cluster_tol = cluster_tol
        self.max_period = max_period
        self.max_merge_iter = max_merge_iter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reconstruct(self, observed_pri: List[float]) -> List[float]:
        """Reconstruct clean PRI sequence from observed (degraded) PRI."""
        values = np.asarray(observed_pri, dtype=np.float64)
        if len(values) == 0:
            return []

        # Step 1: cluster PRI values to find vocabulary
        vocab = self._cluster_values(values)

        # Step 2: quantize observed to vocabulary
        quantized = self._quantize(values, vocab)

        # Step 3: MDL-based dictionary construction → find main pattern
        pattern = self._build_pattern_mdl(quantized)

        # If MDL finds only single-value pattern, try autocorrelation as fallback
        if len(pattern) <= 1:
            period = self._find_period_autocorr(quantized)
            if period > 1:
                pattern = self._extract_pattern_median(values, vocab, period)

        # Step 4: estimate clean sequence length
        target_len = self._estimate_clean_length(values, pattern)

        # Step 5: tile pattern to target length
        clean = self._tile_pattern(pattern, target_len)
        return clean.tolist()

    # ------------------------------------------------------------------
    # Step 1: PRI value clustering (histogram-based)
    # ------------------------------------------------------------------
    def _cluster_values(self, values: np.ndarray) -> np.ndarray:
        """Cluster PRI values into discrete vocabulary centroids."""
        sorted_vals = np.sort(values)
        clusters: List[List[float]] = []
        current: List[float] = [sorted_vals[0]]

        for v in sorted_vals[1:]:
            if abs(v - np.mean(current)) <= self.cluster_tol:
                current.append(v)
            else:
                clusters.append(current)
                current = [v]
        clusters.append(current)

        centroids = np.array([np.median(c) for c in clusters], dtype=np.float64)
        return centroids

    # ------------------------------------------------------------------
    # Step 2: quantise to nearest centroid
    # ------------------------------------------------------------------
    def _quantize(self, values: np.ndarray, vocab: np.ndarray) -> np.ndarray:
        idx = np.argmin(np.abs(values[:, None] - vocab[None, :]), axis=1)
        return vocab[idx]

    # ------------------------------------------------------------------
    # Step 3a: MDL dictionary construction (BPE-style merging)
    # ------------------------------------------------------------------
    def _build_pattern_mdl(self, quantized: np.ndarray) -> np.ndarray:
        """Use MDL principle to discover the dominant PRI pattern.

        Algorithm:
          - Represent sequence as list of symbol-IDs
          - Iteratively merge the most frequent bigram into a new symbol
          - Accept only if total description length decreases
          - After convergence, the most frequent "word" is the PRI pattern
        """
        # symbol → tuple of original PRI values
        unique_vals = sorted(set(quantized.tolist()))
        val_to_sym: dict = {v: i for i, v in enumerate(unique_vals)}
        sym_to_word: dict = {i: (v,) for i, v in enumerate(unique_vals)}
        next_sym = len(unique_vals)

        sequence = [val_to_sym[v] for v in quantized.tolist()]

        for _ in range(self.max_merge_iter):
            if len(sequence) < 2:
                break

            # count bigrams
            bigrams: Counter = Counter()
            for i in range(len(sequence) - 1):
                bigrams[(sequence[i], sequence[i + 1])] += 1
            if not bigrams:
                break

            best_pair, best_count = bigrams.most_common(1)[0]
            if best_count < 2:
                break

            # compute MDL cost change
            old_dict_cost = sum(len(w) for w in sym_to_word.values())
            new_word = sym_to_word[best_pair[0]] + sym_to_word[best_pair[1]]

            # guard: pattern too long → stop
            if len(new_word) > self.max_period:
                break

            new_dict_cost = old_dict_cost + len(new_word)
            n_symbols = len(sym_to_word)

            old_dl = len(sequence) * np.log2(max(n_symbols, 2)) + old_dict_cost
            new_seq_len = len(sequence) - best_count
            new_dl = new_seq_len * np.log2(max(n_symbols + 1, 2)) + new_dict_cost

            if new_dl >= old_dl:
                break  # no improvement

            # accept merge
            new_sym = next_sym
            next_sym += 1
            sym_to_word[new_sym] = new_word

            merged: List[int] = []
            i = 0
            while i < len(sequence):
                if (i < len(sequence) - 1
                        and (sequence[i], sequence[i + 1]) == best_pair):
                    merged.append(new_sym)
                    i += 2
                else:
                    merged.append(sequence[i])
                    i += 1
            sequence = merged

        # Choose dominant word (most frequent in final sequence)
        freq = Counter(sequence)
        best_sym = freq.most_common(1)[0][0]
        pattern_tuple = sym_to_word[best_sym]
        return np.array(pattern_tuple, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 3b: fallback — autocorrelation period detection
    # ------------------------------------------------------------------
    def _find_period_autocorr(self, quantized: np.ndarray) -> int:
        n = len(quantized)
        if n < 4:
            return 1
        max_lag = min(n // 2, self.max_period)
        seq = quantized - np.mean(quantized)
        std = np.std(seq)
        if std < 1e-10:
            return 1
        seq = seq / std

        best_lag, best_corr = 1, -np.inf
        for lag in range(1, max_lag + 1):
            corr = np.mean(seq[:n - lag] * seq[lag:])
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        return best_lag if best_corr > 0.3 else 1

    def _extract_pattern_median(
            self, raw_values: np.ndarray, vocab: np.ndarray, period: int
    ) -> np.ndarray:
        """Extract pattern using per-phase median of raw values, then snap."""
        pattern = np.zeros(period, dtype=np.float64)
        for i in range(period):
            phase_vals = raw_values[i::period]
            med = np.median(phase_vals)
            # snap to nearest vocab entry
            idx = np.argmin(np.abs(vocab - med))
            pattern[i] = vocab[idx]
        return pattern

    # ------------------------------------------------------------------
    # Step 4: estimate clean sequence length
    # ------------------------------------------------------------------
    def _estimate_clean_length(
            self, observed: np.ndarray, pattern: np.ndarray
    ) -> int:
        """Estimate length of the clean PRI sequence.

        Key insight: total time span is preserved across miss / spurious
        degradations since they only shift TOA but don't change overall span.
            sum(observed_PRI) ≈ sum(clean_PRI)
        So: N_clean ≈ sum(observed) / mean(pattern)
        """
        total_time = np.sum(observed)
        mean_pri = np.mean(pattern)
        if mean_pri < 1e-6:
            return len(observed)
        estimated = int(round(total_time / mean_pri))
        return max(estimated, 1)

    # ------------------------------------------------------------------
    # Step 5: tile pattern
    # ------------------------------------------------------------------
    @staticmethod
    def _tile_pattern(pattern: np.ndarray, target_len: int) -> np.ndarray:
        if len(pattern) == 0:
            return np.zeros(target_len)
        repeats = target_len // len(pattern) + 1
        tiled = np.tile(pattern, repeats)
        return tiled[:target_len]
