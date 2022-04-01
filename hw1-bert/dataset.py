from typing import List, Dict
from torch.utils.data import Dataset

import re
import torch
import json

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        label_mapping: Dict[str, int],
        tokenizer,
        max_len: int = 128,
        mode = "train"
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = tokenizer

        """
        data:
            {
                "text": "my check engine light is on and i need to take a look at it",
                "intent": "schedule_maintenance",
                "id": "train-1"
            }
        """

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        intent, ids = [], []
        for sample in samples:
            ids.append(sample["id"])
            if self.mode != "test":
                intent.append(self.label_mapping[sample["intent"]])

        inputs = self.tokenizer([data["text"] for data in samples], padding = "max_length", truncation = True, return_tensors = "pt")
        if self.mode != "test":
            intent = torch.tensor(intent) #, dtype = torch.float)
            return inputs, intent, ids
        return inputs, ids

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class STDataset(Dataset):
    """
        Slot Tagging Dataset
    """
    def __init__(
        self,
        data_path: str,
        label_mapping: Dict[str, int],
        tokenizer,
        max_len: int = 128,
        mode = "train"
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # print(self.label_mapping)
        tags, length, ids = [], [], []
        for sample in samples:
            ids.append(sample["id"])
            if self.mode != "test":
                tags.append([self.label_mapping[tag] for tag in sample["tags"]])
            length.append(len(sample["tokens"]))

        inputs = self.tokenizer([" ".join(data["tokens"]) for data in samples], padding = "max_length", truncation = True, return_tensors = "pt")
 
        if self.mode != "test":
            tags = torch.tensor(tags) #, dtype = torch.float)
            return inputs, tags, length, ids
        return inputs, length, ids


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
