from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
import numpy as np


@dataclass
class QuantizerConfig:
    """
    PRI tokenizer config.

    mode:
        - 'prototype': build codebook from known clean PRI prototypes.
        - 'uniform': build a compact codebook with integer keys 1..N, where N=num_bins.

    For uniform mode, each key corresponds to one uniformly spaced PRI value in
    [min_value, max_value].
    """

    mode: str = 'prototype'

    # prototype mode
    prototypes: Optional[Sequence[float]] = None
    prototype_from_sequences: Optional[Sequence[Sequence[float]]] = None
    sort_prototypes: bool = True
    deduplicate_tol: float = 1e-8

    # uniform mode
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    num_bins: Optional[int] = None
    key_start: int = 1

    # shared
    snap_tolerance: Optional[float] = None
    add_special_tokens: bool = True


class PRIQuantizer:
    """
    Encode continuous PRI values into discrete token ids.

    Token layout (default):
      0: [START]
      1: [END]
      2: [UNK]
      3: [PAD]
      4: [SEP]
      5.. : quantized PRI keys / prototypes

    IMPORTANT:
      [SEP] must be independent from [END]. The original code reused [END] as
      [SEP], which makes the source-target boundary ambiguous and significantly
      harms seq2seq diffusion training.
    """

    def __init__(self, config: QuantizerConfig):
        self.config = config
        self.start_token_id = 0
        self.end_token_id = 1
        self.unk_token_id = 2
        self.pad_token_id = 3
        self.sep_token_id = 4
        self.offset = 5 if config.add_special_tokens else 0
        self.special_token_ids = {
            self.start_token_id,
            self.end_token_id,
            self.unk_token_id,
            self.pad_token_id,
            self.sep_token_id,
        }

        codebook = self._build_codebook(config)
        if codebook.size == 0:
            raise ValueError('Quantizer codebook is empty.')

        self.prototypes = codebook.astype(np.float32)
        self.vocab_size = int(len(self.prototypes) + self.offset)
        self.num_pri_tokens = int(len(self.prototypes))
        self.key_start = int(config.key_start)
        self.key_end = self.key_start + self.num_pri_tokens - 1

    def _build_codebook(self, config: QuantizerConfig) -> np.ndarray:
        mode = config.mode.lower().strip()
        if mode == 'uniform':
            if config.num_bins is None or int(config.num_bins) <= 0:
                raise ValueError('uniform quantizer requires num_bins > 0.')
            if config.min_value is None or config.max_value is None:
                raise ValueError('uniform quantizer requires min_value and max_value.')

            min_value = float(config.min_value)
            max_value = float(config.max_value)
            num_bins = int(config.num_bins)

            if max_value < min_value:
                raise ValueError('max_value must be >= min_value.')
            if max_value == min_value:
                return np.asarray([min_value] * num_bins, dtype=np.float32)
            return np.linspace(min_value, max_value, num_bins, dtype=np.float32)

        if mode != 'prototype':
            raise ValueError(f'Unsupported quantizer mode: {config.mode}')

        values: List[float] = []
        if config.prototypes is not None:
            values.extend(float(v) for v in config.prototypes)
        if (not values) and config.prototype_from_sequences is not None:
            for seq in config.prototype_from_sequences:
                values.extend(float(v) for v in seq)

        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr

        if config.sort_prototypes:
            arr = np.sort(arr)

        deduped: List[float] = []
        for v in arr.tolist():
            if not deduped or abs(v - deduped[-1]) > config.deduplicate_tol:
                deduped.append(v)
        return np.asarray(deduped, dtype=np.float32)

    def _nearest_prototype_index(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 0:
            values = values.reshape(1)

        right = np.searchsorted(self.prototypes, values, side='left')
        right = np.clip(right, 0, len(self.prototypes) - 1)
        left = np.clip(right - 1, 0, len(self.prototypes) - 1)

        left_dist = np.abs(values - self.prototypes[left])
        right_dist = np.abs(values - self.prototypes[right])
        choose_right = right_dist < left_dist
        idx = np.where(choose_right, right, left).astype(np.int64)

        if self.config.snap_tolerance is not None:
            min_dist = np.minimum(left_dist, right_dist)
            idx[min_dist > self.config.snap_tolerance] = -1
        return idx

    def _value_to_token_ids(self, values: np.ndarray) -> np.ndarray:
        proto_idx = self._nearest_prototype_index(values)
        token_ids = np.where(proto_idx >= 0, proto_idx + self.offset, self.unk_token_id)
        return token_ids.astype(np.int64)

    def _token_ids_to_values(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        valid = (ids >= self.offset) & (ids < self.vocab_size)
        out = np.full(ids.shape, np.nan, dtype=np.float32)
        out[valid] = self.prototypes[ids[valid] - self.offset]
        return out

    def token_id_to_key(self, token_id: int) -> Optional[int]:
        if token_id < self.offset or token_id >= self.vocab_size:
            return None
        return int(token_id - self.offset + self.key_start)

    def key_to_token_id(self, key: int) -> Optional[int]:
        key = int(key)
        if key < self.key_start or key > self.key_end:
            return None
        return int(key - self.key_start + self.offset)

    def value_to_key(self, value: float) -> int:
        idx = int(self._nearest_prototype_index(np.asarray([value], dtype=np.float32))[0])
        if idx < 0:
            return self.key_start
        return idx + self.key_start

    def key_to_value(self, key: int) -> float:
        token_id = self.key_to_token_id(key)
        if token_id is None:
            raise ValueError(f'Key {key} is out of range [{self.key_start}, {self.key_end}]')
        return float(self.prototypes[token_id - self.offset])

    def values_to_keys(self, values: Sequence[float]) -> List[int]:
        idx = self._nearest_prototype_index(np.asarray(values, dtype=np.float32))
        return (idx + self.key_start).astype(np.int64).tolist()

    def keys_to_values(self, keys: Sequence[int]) -> List[float]:
        out: List[float] = []
        for key in keys:
            token_id = self.key_to_token_id(key)
            if token_id is None:
                continue
            out.append(float(self.prototypes[token_id - self.offset]))
        return out

    def encode_values(self, values: Sequence[float], add_boundary_tokens: bool = True) -> List[int]:
        arr = np.asarray(values, dtype=np.float32)
        ids = self._value_to_token_ids(arr).tolist()
        if add_boundary_tokens and self.config.add_special_tokens:
            ids = [self.start_token_id] + ids + [self.end_token_id]
        return ids

    def decode_ids(self, ids: Sequence[int], remove_special: bool = True) -> List[float]:
        out: List[float] = []
        for token_id in ids:
            if remove_special and token_id in self.special_token_ids:
                continue
            if token_id < self.offset or token_id >= self.vocab_size:
                continue
            out.append(float(self.prototypes[token_id - self.offset]))
        return out

    def decode_ids_to_keys(self, ids: Sequence[int], remove_special: bool = True) -> List[int]:
        out: List[int] = []
        for token_id in ids:
            if remove_special and token_id in self.special_token_ids:
                continue
            key = self.token_id_to_key(int(token_id))
            if key is not None:
                out.append(key)
        return out

    def nearest_token_from_values(self, values: Sequence[float]) -> List[int]:
        return self.encode_values(values, add_boundary_tokens=False)

    def encode_batch(self, batch_values: Iterable[Sequence[float]], add_boundary_tokens: bool = True) -> List[List[int]]:
        return [self.encode_values(v, add_boundary_tokens=add_boundary_tokens) for v in batch_values]

    def get_prototype_id_to_value_map(self) -> List[float]:
        return self.prototypes.tolist()

    def get_key_value_pairs(self) -> List[tuple[int, float]]:
        return [(self.key_start + i, float(v)) for i, v in enumerate(self.prototypes.tolist())]

    @classmethod
    def from_ideal_pri_sequences(
        cls,
        ideal_sequences: Sequence[Sequence[float]],
        *,
        add_special_tokens: bool = True,
        snap_tolerance: Optional[float] = None,
        sort_prototypes: bool = True,
        deduplicate_tol: float = 1e-6,
    ) -> 'PRIQuantizer':
        cfg = QuantizerConfig(
            mode='prototype',
            prototype_from_sequences=ideal_sequences,
            add_special_tokens=add_special_tokens,
            snap_tolerance=snap_tolerance,
            sort_prototypes=sort_prototypes,
            deduplicate_tol=deduplicate_tol,
        )
        return cls(cfg)
