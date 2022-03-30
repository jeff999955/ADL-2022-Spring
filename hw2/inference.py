import os
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from dataset import *
from torch.utils.data import DataLoader
from accelerate import Accelerator

from tqdm import tqdm

from utils import same_seeds
from model import *

import wandb

import numpy as np
import collections


@torch.no_grad()
def mc_predict(data_loader, model):
    model.eval()
    relevant = {}
    for batch in tqdm(data_loader):
        ids, input_ids, attention_masks, token_type_ids, labels = batch
        output = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        pred = output.logits.argmax(dim=-1).cpu().numpy()
        for _id, _pred in zip(ids, pred):
            relevant[_id] = int(_pred)

    return relevant


@torch.no_grad()
def qa_predict(args, data_loader, model, n_best=1):
    ret = []
    model.eval()
    for batch in tqdm(data_loader):
        answers = []

        ids, inputs = batch
        context = inputs["context"][0]
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        qa_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        start_logits = qa_output.start_logits.cpu().numpy()
        end_logits = qa_output.end_logits.cpu().numpy()
        for i in range(len(input_ids)):
            start_logit = start_logits[i]
            end_logit = end_logits[i]
            offsets = inputs["offset_mapping"][i]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue

                    answers.append(
                        {
                            "text": context[
                                offsets[start_index][0] : offsets[end_index][1]
                            ],
                            "logit_score": start_logit[start_index]
                            + end_logit[end_index],
                        }
                    )
        best_answer = max(answers, key=lambda x: x["logit_score"])
        ret.append((ids[0], best_answer["text"]))
    return ret


def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)

    ckpt = torch.load(os.path.join(args.mc_ckpt))
    namae = ckpt["name"]
    config = AutoConfig.from_pretrained(namae)
    tokenizer = AutoTokenizer.from_pretrained(
        namae, config=config, model_max_length=args.max_len, use_fast=True
    )
    model = MultipleChoiceModel(args, config, ckpt["name"])
    model.load_state_dict(ckpt["model"])
    test_set = MultipleChoiceDataset(args, tokenizer, mode="test")
    test_loader = DataLoader(
        test_set,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        batch_size=1,
    )
    model, test_loader = accelerator.prepare(model, test_loader)
    relevant = mc_predict(test_loader, model)

    del ckpt, config, tokenizer, model, test_loader
    torch.cuda.empty_cache()

    ckpt = torch.load(os.path.join(args.qa_ckpt))
    namae = ckpt["name"]
    config = AutoConfig.from_pretrained(namae)
    tokenizer = AutoTokenizer.from_pretrained(
        namae, config=config, model_max_length=args.max_len, use_fast=True
    )
    model = QuestionAnsweringModel(args, config, ckpt["name"])
    model.load_state_dict(ckpt["model"])
    test_set = QuestionAnsweringDataset(args, tokenizer, mode="test", relevant=relevant)
    test_loader = DataLoader(
        test_set,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        batch_size=1,
    )
    model, test_loader = accelerator.prepare(model, test_loader)

    answers = qa_predict(args, test_loader, model, n_best=20)
    with open(args.csv_path, "w") as f:
        print("id,answer", file=f)
        for _id, answer in answers:
            if "「" in answer and "」" not in answer:
                answer += "」"
            elif "「" not in answer and "」" in answer:
                answer = "「" + answer
            if "《" in answer and "》" not in answer:
                answer += "》"
            elif "《" not in answer and "》" in answer:
                answer = "《" + answer
            answer = answer.replace(",", "")
            print(f"{_id},{answer}", file=f)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to save the cache file.",
        default="./cache",
    )

    parser.add_argument("--context_path", type=Path, default="./context.json")
    parser.add_argument("--json_path", type=Path, default="./test.json")
    parser.add_argument("--mc_ckpt", type=Path, default="./mc.ckpt")

    parser.add_argument("--qa_ckpt", type=Path, default="./qa.ckpt")
    parser.add_argument("--csv_path", type=str, default="submission.csv")

    parser.add_argument("--max_len", type=int, default=512)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
