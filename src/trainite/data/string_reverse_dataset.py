import random
import string
import uuid
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from .tokenizer import encode, TOKEN_MAPPING, PAD_ID
from ..config import DataConfig


class StringReversalDataset(Dataset):
    def __init__(self, min_len: int = 5, max_len: int = 20, num_samples: int = 1000, token_mapping: dict[str, int] = TOKEN_MAPPING):
        self.min_len = min_len
        self.max_len = max_len
        self.num_samples = num_samples
        self.token_mapping = token_mapping
        self.all_samples = self._generate_data()

    def _generate_data(self) -> list[dict[str, str]]:
        data = []
        for _ in range(self.num_samples):
            str_len = random.randint(self.min_len, self.max_len)
            rand_str = ''.join(random.choices(string.ascii_lowercase, k=str_len))
            sample_id = uuid.uuid4().hex
            data.append({'id': sample_id, 'input': rand_str, 'output': rand_str[::-1]})
        return data
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        datapoint = self.all_samples[idx]

        input_ids = encode(datapoint["input"], self.token_mapping)
        output_ids = encode(datapoint["output"], self.token_mapping)

        full_sequence = input_ids + [self.token_mapping["<SEP>"]] + output_ids + [self.token_mapping["<EOS>"]]

        loss_mask = [0] * len(full_sequence)
        seg_start = len(input_ids) + 1
        seg_end = len(input_ids) + 1 + len(output_ids) + 1

        for i in range(seg_start, seg_end):
            loss_mask[i] = 1

        return {
            "input_ids": torch.tensor(full_sequence, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "id": datapoint["id"]
        }
    
    def __len__(self) -> int:
        return len(self.all_samples)
    
    
def collate_fn(batch: list[dict], pad_id: int = PAD_ID) -> dict[str, Any]:
    input_ids_lst = [item["input_ids"] for item in batch]
    loss_masks_lst = [item["loss_mask"] for item in batch]
    ids_lst = [item["id"] for item in batch]

    padded_input_ids = pad_sequence(input_ids_lst, batch_first=True, padding_value=pad_id)
    padded_loss_masks = pad_sequence(loss_masks_lst, batch_first=True, padding_value=0.0)

    return {
        "input_ids": padded_input_ids,
        "loss_mask": padded_loss_masks,
        "ids": ids_lst
    }

def build_dataloaders(
    data_cfg: DataConfig,
    token_mapping: dict[str, int] = TOKEN_MAPPING,
) -> tuple[
    Subset,
    Subset,
    Subset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    total = data_cfg.train_samples + data_cfg.val_samples + data_cfg.test_samples

    full_dataset = StringReversalDataset(
        min_len=data_cfg.min_str_len,
        max_len=data_cfg.max_str_len,
        num_samples=total,
        token_mapping=token_mapping
    )

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [data_cfg.train_samples, data_cfg.val_samples, data_cfg.test_samples],
    )

    collated_with_pad = partial(collate_fn, pad_id=token_mapping["<PAD>"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=collated_with_pad,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        collate_fn=collated_with_pad,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        collate_fn=collated_with_pad,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader