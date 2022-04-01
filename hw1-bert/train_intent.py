import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import trange

from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from utils import  same_seeds
from model import IntentClassifier

from tqdm import tqdm
import math
import os

from accelerate import Accelerator

TRAIN = "train"
VAL = "eval"
TEST = "test"
SPLITS = [TRAIN, VAL, TEST]


def train(model, data_loader, optimizer, scheduler = None):
    train_loss = []
    train_accs = []

    model.train()
    for inputs, intent, ids in tqdm(data_loader):
        outputs = model(**inputs, labels = intent)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        acc = (logits.argmax(dim=-1) == intent).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    return train_loss, train_acc


@torch.no_grad()
def validate(model, data_loader):
    model.eval()
    valid_loss = []
    valid_accs = []

    for inputs, intent, ids in tqdm(data_loader):
        outputs = model(**inputs, labels = intent)
        loss = outputs.loss
        logits = outputs.logits
        acc = (logits.argmax(dim=-1) == intent).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    return valid_loss, valid_acc


def main(args):
    same_seeds(args.seed)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    
    config = AutoConfig.from_pretrained(args.model_name, num_labels = 150)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, config=config, model_max_length=args.max_len, use_fast=True)

    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, intent2idx, tokenizer, max_len = args.max_len, mode = split)
        for split, split_data in data_paths.items()
    }

    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, collate_fn=datasets[TRAIN].collate_fn)
    valid_loader = DataLoader(datasets[VAL], batch_size=args.batch_size,
                              shuffle=False, pin_memory=True, collate_fn=datasets[VAL].collate_fn)
    # TODO: crecate DataLoader for train / dev datasets
    model = IntentClassifier(config)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-5, weight_decay = 1e-6)
    total_step = args.num_epoch * (len(train_loader) // args.batch_size)
    warmup_step = int(total_step) * 0.1
    scheduler = None # get_cosine_schedule_with_warmup(optimizer, warmup_step, total_step - warmup_step)
    accelerator = Accelerator()
    model, train_loader, valid_loader, optimizer = accelerator.prepare(
    model, train_loader, valid_loader, optimizer 
    )
    best_loss = 1268
    best_acc = -1
    for epoch in range(1, args.num_epoch + 1):
        print(f"Epoch {epoch}:")
        train_loss, train_acc = train(
            model, train_loader, optimizer, scheduler)
        valid_loss, valid_acc = validate(model, valid_loader)
        print(f"Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
        print(f"Valid Accuracy: {valid_acc:.4f}, Valid Loss: {valid_loss:.4f}")




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1268)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--wd", type=float, default=1e-6)

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument("--num_epoch", type=int, default=5)

    parser.add_argument("--model_name", type = str, default = "hfl/chinese-macbert-base")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    main(args)
