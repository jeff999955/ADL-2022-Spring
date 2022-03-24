import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
from torch import nn, autocast
from transformers import AutoConfig, AutoTokenizer
from dataset import MultipleChoiceDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from tqdm import tqdm

from utils import same_seeds
from model import MultipleChoiceModel

import wandb

log_time = 3

def train(args, data_loader, model, optimizer, scaler):
    global log_time
    train_loss = []
    train_accs = []

    model.train()

    for idx, batch in enumerate(tqdm(data_loader)):
        ids, input_ids, attention_masks, token_type_ids, labels = batch
        with autocast(device_type = "cpu" if args.device == "cpu" else "cuda"):
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids,
                labels = labels
            )
            acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
            loss = loss / args.accu_step
        
        scaler.scale(loss).backward()

        if ((idx + 1) % args.accu_step == 0) or (idx == len(data_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    return train_loss, train_acc


@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(data_loader):
        ids, input_ids, attention_masks, token_type_ids, labels = batch
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    return valid_loss, valid_acc


def main(args):
    same_seeds(args.seed)

    config = AutoConfig.from_pretrained(args.model_name, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, config=config, model_max_length=512, use_fast=True
    )
    model = MultipleChoiceModel(args, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    scaler = GradScaler()

    starting_epoch = 1
    if args.pretrain:
        print(f'loading model from {args.pretrain}')
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        starting_epoch = ckpt['epoch']
    if args.wandb:
        wandb.watch(model)

    train_set = MultipleChoiceDataset(args, tokenizer)
    valid_set = MultipleChoiceDataset(args, tokenizer, mode = "valid")

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


    best_loss = float("inf")

    for epoch in range(starting_epoch, args.num_epoch + 1):
        print(f"Epoch {epoch}:")
        train_loss, train_acc = train(args, train_loader, model, optimizer, scaler)
        valid_loss, valid_acc = validate(valid_loader, model)
        print(f"Train Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}")
        print(f"Valid Accuracy: {valid_acc:.2f}, Valid Loss: {valid_loss:.2f}")
        if args.wandb:
            wandb.log(
                {
                    "Train Accuracy": train_acc,
                    "Train Loss": train_loss,
                    "Validation Accuracy": valid_acc,
                    "Validation Loss": valid_loss,
                }
            )
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {
                    "name": args.model_name,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"{args.prefix}mc_loss.ckpt"),
            )


def parse_args():
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
        default="./ckpt/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=1e-6)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--accu_step", type = int, default = 8)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--pretrain", type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    if args.wandb:
        wandb.login()
        wandb.init(project="Multiple Choice")
        wandb.config.update(args)
    main(args)
