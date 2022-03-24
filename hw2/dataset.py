import os
import json
import torch
import random
from torch.utils.data import Dataset

from tqdm import tqdm


class MultipleChoiceDataset(Dataset):
    def __init__(self, args, tokenizer, mode = "train"):
        self.mode = mode
        self.device = args.device
        self.json_data = []
        try:
            self.json_data = torch.load(os.path.join(args.cache_dir, f"mc_{mode}.dat"))
        except Exception as e:
            print(e)
            with open(os.path.join(args.data_dir, "context.json"), "r") as f:
                context_data = json.load(f)
            with open(os.path.join(args.data_dir, f"{mode}.json"), "r") as f:
                json_data = json.load(f)
                print("Preprocessing Data:")
                for data in tqdm(json_data):
                    if mode != 'test':
                        label = data["paragraphs"].index(data["relevant"])
                    else:
                        label = random.choice(list(range(len(data["paragraphs"]))))
                    
                    qa_pair = [
                        "{} {}".format(data["question"], context_data[i])
                        for i in data["paragraphs"]
                    ]
                    features = tokenizer(
                        qa_pair, padding="max_length", truncation=True, return_tensors="pt"
                    )
                    
                    self.json_data.append(
                        {
                            "id": data["id"],
                            "input_ids": features["input_ids"],
                            "token_type_ids": features["token_type_ids"],
                            "attention_mask": features["attention_mask"],
                            "label": label
                        }
                    )
            torch.save(self.json_data, os.path.join(args.cache_dir, f"mc_{mode}.dat"))

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        return self.json_data[idx]

    def collate_fn(self, batch):
        ids, input_ids, attention_masks, token_type_ids, labels = [], [], [], [], []
        for sample in batch:
            ids.append(sample["id"])
            input_ids.append(sample["input_ids"])
            token_type_ids.append(sample["token_type_ids"])
            attention_masks.append(sample["attention_mask"])
            labels.append(sample["label"])
        try:
            input_ids = torch.stack(input_ids).to(self.device)
            attention_masks = torch.stack(attention_masks).to(self.device)
            token_type_ids = torch.stack(token_type_ids).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)
        except Exception as e:
            print(e)
            print(ids)
        return ids, input_ids, attention_masks, token_type_ids, labels

def parse_args():
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default=".",
    )
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to save the cache file.",
        default="./cache",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument(
        "--recurrent", type=str, choices=["rnn", "lstm", "gru"], default="gru"
    )
    parser.add_argument("--freeze", type=bool, default=False)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=1e-6)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=200)

    parser.add_argument("--accu_step", type = int, default = 8)

    parser.add_argument("--prefix", type=str, default="")

    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice
    from torch.utils.data import DataLoader

    from tqdm import tqdm
    from pathlib import Path
    args = parse_args()
    
    config = AutoConfig.from_pretrained(args.model_name, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, config=config, model_max_length=512, use_fast=True
    )
    
    
    train_set = MultipleChoiceDataset(args, tokenizer)
    valid_set = MultipleChoiceDataset(args, tokenizer, mode = "valid")
    test_set = MultipleChoiceDataset(args, tokenizer, mode = "test")
    
    train_loader = DataLoader(
        train_set,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=valid_set.collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        test_set,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
    )
    
    for data in tqdm(train_loader):
        pass
    for data in tqdm(valid_loader):
        pass
    for data in tqdm(test_loader):
        pass
            
