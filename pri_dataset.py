from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from .pri_tokenizer import PRIQuantizer, QuantizerConfig
except ImportError:
    from pri_tokenizer import PRIQuantizer, QuantizerConfig


@dataclass
class PRISample:
    observed_pri: Sequence[float]
    clean_pri: Sequence[float]


class PRIDiffuSeqDataset(Dataset):
    """
    Build DiffuSeq-style seq2seq training examples.

    Layout:
      input_ids = [src_tokens] + [SEP] + [trg_tokens] + [PAD...]
      input_mask:
        0 for source / separator / pad
        1 only for true target tokens

    Why pad must be 0 here:
      The original implementation marked pad as 1, which forced the diffusion
      loss and CE loss to optimize many padded positions. This makes the model
      collapse toward predicting PAD while real PRI token accuracy stays low.
    """

    def __init__(self, samples: Sequence[PRISample], quantizer: PRIQuantizer, seq_len: int):
        self.samples = list(samples)
        self.quantizer = quantizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def _build_item(self, sample: PRISample) -> Dict[str, torch.Tensor]:
        src = self.quantizer.encode_values(sample.observed_pri, add_boundary_tokens=True)
        trg = self.quantizer.encode_values(sample.clean_pri, add_boundary_tokens=True)

        src_body = src[:-1]
        trg_body = trg[:-1]
        while len(src_body) + len(trg_body) > self.seq_len - 3:
            if len(src_body) >= len(trg_body):
                src_body.pop()
            else:
                trg_body.pop()

        src_body.append(self.quantizer.end_token_id)
        trg_body.append(self.quantizer.end_token_id)

        ids = src_body + [self.quantizer.sep_token_id] + trg_body
        src_plus_sep_len = len(src_body) + 1
        target_len = len(trg_body)

        if len(ids) < self.seq_len:
            pad_n = self.seq_len - len(ids)
            ids = ids + [self.quantizer.pad_token_id] * pad_n
        else:
            ids = ids[: self.seq_len]
            pad_n = 0

        target_mask = [0] * src_plus_sep_len + [1] * target_len
        if len(target_mask) < self.seq_len:
            target_mask = target_mask + [0] * (self.seq_len - len(target_mask))
        else:
            target_mask = target_mask[: self.seq_len]

        attention_mask = [1] * (self.seq_len - pad_n) + [0] * pad_n
        attention_mask = attention_mask[: self.seq_len]

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'input_mask': torch.tensor(target_mask, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._build_item(self.samples[idx])


class PRICollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        out = {
            'input_ids': torch.stack([x['input_ids'] for x in batch], dim=0),
            'input_mask': torch.stack([x['input_mask'] for x in batch], dim=0),
        }
        if 'attention_mask' in batch[0]:
            out['attention_mask'] = torch.stack([x['attention_mask'] for x in batch], dim=0)
        return out
