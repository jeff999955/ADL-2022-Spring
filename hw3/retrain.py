import os
from argparse import ArgumentParser
from pathlib import Path
from torch.distributions import Categorical

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb

from tw_rouge import get_rouge

from utils import *


def rl_loss(args, accelerator, tokenizer, model, batch, logits):
    generate_tokens = accelerator.unwrap_model(model).generate(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_length=args.max_answer_len,
    )
    generate_tokens = accelerator.pad_across_processes(
        generate_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    generate_tokens = accelerator.gather(generate_tokens)
    generate_preds = tokenizer.batch_decode(
        generate_tokens.cpu().numpy(), skip_special_tokens=True
    )

    sample_tokens = accelerator.unwrap_model(model).generate(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_length=args.max_answer_len,
        do_sample=True,
        top_k=5,
        top_p=0.1,
    )
    sample_tokens = accelerator.pad_across_processes(
        sample_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    sample_tokens = accelerator.gather(sample_tokens)
    sample_preds = tokenizer.batch_decode(
        sample_tokens.cpu().numpy(), skip_special_tokens=True
    )

    labels = batch["labels"]
    labels = accelerator.pad_across_processes(
        batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
    )
    labels = accelerator.gather(labels).cpu().numpy()
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    generate_preds, labels = postprocess_text(generate_preds, labels)
    sample_preds, _ = postprocess_text(sample_preds, [])
    ws = (
        lambda scores: scores["rouge-1"]["f"] * 0.3
        + scores["rouge-2"]["f"] * 0.5
        + scores["rouge-l"]["f"] * 0.2
    )
    generate_scores = get_rouge(generate_preds, labels, avg=False)
    generate_rewards = torch.tensor([ws(scores) for scores in generate_scores])
    sample_scores = get_rouge(sample_preds, labels, avg=False)
    sample_rewards = torch.tensor([ws(scores) for scores in sample_scores])

    criterion = torch.nn.CrossEntropyLoss()
    N, Lg, Ls, C = (
        logits.shape[0],
        generate_tokens.shape[1],
        sample_tokens.shape[1],
        logits.shape[-1],
    )
    loss_input = logits[:, :Ls, :].reshape(N * Ls, C)
    sample_probs = criterion(loss_input, sample_tokens.view(-1))
    # print(sample_probs)
    # print(sample_probs.shape)
    diff_rewards = (sample_rewards - generate_rewards).to(accelerator.device)
    rl_loss = (diff_rewards * sample_probs).mean()

    return rl_loss


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument("--model_name", type=str, default="google/mt5-small")

    # data
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--max_answer_len", type=int, default=64)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--scheduler", type=str, default="cosine")

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--accu_step", type=int, default=8)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()
    return args


def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    print(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.ckpt_dir)

    model.resize_token_embeddings(len(tokenizer))

    raw_dataset = load_dataset(
        "json", data_files={"train": os.path.join(args.data_dir, "train.jsonl")}
    )
    cols = raw_dataset["train"].column_names
    train_prep = preprocess_function(tokenizer=tokenizer, args=args)
    dataset = raw_dataset.map(
        train_prep, batched=True, keep_in_memory=True, num_proc=8, remove_columns=cols
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_loader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    if args.wandb:
        wandb.watch(model)
    best_loss = np.inf
    for epoch in range(args.num_epoch):
        print(f"epoch {epoch}:")
        train_loss = []

        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            outputs = model(**batch)
            loss = rl_loss(args, accelerator, tokenizer, model, batch, outputs.logits)
            print(loss)
            train_loss.append(loss.item())
            loss = loss / args.accu_step

            accelerator.backward(loss)
            if ((step + 1) % args.accu_step == 0) or (step == len(train_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()
        train_loss = np.mean(train_loss)
        print(f"train loss: {train_loss:.4f}")
        # evaluation
        if args.wandb:
            wandb.log(
                {
                    "train loss": train_loss,
                }
            )

        if train_loss < best_loss:
            best_loss = train_loss
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.ckpt_dir, save_function=accelerator.save
            )
            tokenizer.save_pretrained(args.ckpt_dir)


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    if args.wandb:
        wandb.login()
        wandb.init(project="News Summarization")
        wandb.config.update(args)
    main(args)
