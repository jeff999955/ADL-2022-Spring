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
        max_len: int,
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(data_path)
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode

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
        text, intent, id = [], [], []
        for sample in samples:
            id.append(sample["id"])
            if self.mode != "test":
                intent.append(self.label_mapping[sample["intent"]])
            text.append(re.sub(r'[^\w\s]', '', sample["text"]).split())

        # padding is done by TA SeemsGood
        text = self.vocab.encode_batch(text, self.max_len)
        text = torch.tensor(text)
        if self.mode != "test":
            intent = torch.tensor(intent)
            return text, intent, id
        return text, id

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
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        mode: str = "train"
    ):
        self.data = data
        self.vocab: Vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.mode = mode

        

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
        tokens, tags, length, id = [], [], [], []
        for sample in samples:
            id.append(sample["id"])
            if self.mode != "test":
                tags.append([self.label_mapping[tag] for tag in sample["tags"]])
            length.append(len(sample["tokens"]))
            tokens.append(sample["tokens"])

        # padding is done by TA SeemsGood
        tokens = self.vocab.encode_batch(tokens, self.max_len)
        tokens = torch.tensor(tokens)
        if self.mode != "test":
            tags = pad_to_len(tags, self.max_len, 9)
            tags = torch.tensor(tags)
            # print(tags)
            return tokens, tags, length, id
        return tokens, length, id


    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
