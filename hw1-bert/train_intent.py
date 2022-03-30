import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import trange

from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from utils import Vocab, same_seeds
from model import IntentClassifier

from tqdm import tqdm
import math
import os

import wandb

TRAIN = "train"
VAL = "eval"
TEST = "test"
SPLITS = [TRAIN, VAL, TEST]


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
        transformer.get_cosine_schedule_with_warmup
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train(model, device, data_loader, criterion, optimizer):
    train_loss = []
    train_accs = []

    model.train()
    for text, intent, id in tqdm(data_loader):
        text, intent = text.to(device), intent.to(device)
        logits = model(text)
        loss = criterion(logits, intent)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()

        acc = (logits.argmax(dim=-1) == intent).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    return train_loss, train_acc


@torch.no_grad()
def validate(model, device, data_loader, criterion):
    model.eval()
    valid_loss = []
    valid_accs = []

    for text, intent, id in tqdm(data_loader):
        text, intent = text.to(device), intent.to(device)
        logits = model(text)
        loss = criterion(logits, intent)
        acc = (logits.argmax(dim=-1) == intent).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    return valid_loss, valid_acc


def main(args):
    same_seeds(args.seed)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}

    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_loader = DataLoader(datasets[TRAIN], batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, collate_fn=datasets[TRAIN].collate_fn)
    valid_loader = DataLoader(datasets[VAL], batch_size=args.batch_size,
                              shuffle=False, pin_memory=True, collate_fn=datasets[VAL].collate_fn)
    # TODO: crecate DataLoader for train / dev datasets

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = IntentClassifier(embeddings, args.hidden_size, args.num_layers, recurrent=args.recurrent,
                             dropout=args.dropout, bidirectional=args.bidirectional, freeze = args.freeze, prefix=args.prefix).to(args.device)
    print(model)

    # TODO: init optimizer
    embedding_param, model_param = [], []
    for name, param in model.named_parameters():
        if "embedding" in name.lower():
            embedding_param.append(param)
        else:
            model_param.append(param)
    optimizer = torch.optim.Adam(
        [{'params':model_param}, {'params':embedding_param}], lr=args.lr, weight_decay=args.wd)
    optimizer.param_groups[0]['lr'] = args.lr
    optimizer.param_groups[1]['lr'] = 1e-5
    criterion = nn.CrossEntropyLoss()
    total_step = math.ceil(
        len(datasets[TRAIN]) // args.batch_size) * args.num_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.num_epoch)
    wandb.watch(model)
    # 
    best_loss = 1268
    best_acc = -1
    for epoch in range(1, args.num_epoch + 1):
        train_loss, train_acc = train(
            model, args.device, train_loader, criterion, optimizer)
        valid_loss, valid_acc = validate(
            model, args.device, valid_loader, criterion)
        wandb.log({'Train Accuracy' : train_acc, 'Train Loss': train_loss, "Validation Accuracy": valid_acc, "Validation Loss": valid_loss})
        if scheduler is not None:
            scheduler.step(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(args.ckpt_dir, f"{model.name}_loss.ckpt"))
        elif valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(args.ckpt_dir, f"{model.name}_acc.ckpt"))
        elif epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(args.ckpt_dir, f"{model.name}_{epoch}.ckpt"))




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
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--recurrent", type=str,
                        choices=["rnn", "lstm", "gru"], default="gru")
    parser.add_argument("--freeze", type = bool, default = False)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=1e-6)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    parser.add_argument("--prefix", type = str, default = "")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    wandb.login()
    wandb.init(project = "intent classification")
    wandb.config.update(args)
    main(args)
