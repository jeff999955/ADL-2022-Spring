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

    ret = {}

    for batch in tqdm(data_loader):
        ids, input_ids, attention_masks, token_type_ids, _ = batch
        output = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )

        pred = output.logits.argmax(dim = -1)

        for _id, _pred in zip(ids, pred):
            ret[_id] = _pred
    return ret

@torch.no_grad()
def qa_predict(data_loader, model, n_best = 20, max_answer_length = 30):
    model.eval()

    ret = []
    for ids, inputs in tqdm(data_loader):
        answer = []
        contex = inputs["context"][0]

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        qa_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        start_logit = qa_output.start_logits.cpu().numpy()
        end_logit = qa_output.end_logits.cpu().numpy()
        offsets = inputs["offset_mapping"]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

        best_answer = max(answers, key=lambda x: x["logit_score"])
        ret.append((ids[0], best_answer["text"]))
    return ret
     


def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    tags = ["mc", "qa"]
    ckpt, config, tokenizer = {}, {}, {}
    for tag in tags:
        ckpt[tag] = torch.load(os.path.join(args.ckpt_dir, f"{tag}.ckpt"))
        namae = ckpt[tag]["name"] 
        config[tag] = AutoConfig.from_pretrained(namae)
        tokenizer[tag] = AutoTokenizer.from_pretrained(
            namae, config=config[tag], model_max_length=args.max_len, use_fast=True
        )

    model = MultipleChoiceModel(args, config["mc"], ckpt["mc"]["name"])
    test_set = MultipleChoiceDataset(args, tokenizer["mc"], mode="test")
    test_loader = DataLoader(
        test_set,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        batch_size=1,
    )
    model, test_loader = accelerator.prepare(
            model, test_loader
    )
    relevant = mc_predict(test_loader, model["mc"])

    model = QuestionAnsweringModel(args, config["qa"], ckpt["qa"]["name"])
    test_set = QuestionAnsweringDataset(args, tokenizer["qa"], mode="test")
    test_loader = DataLoader(
        test_set,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        batch_size=1,
    )
    model, test_loader = accelerator.prepare(
            model, test_loader
    )
    ansaa = qa_predict(test_loader, model)


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
    parser.add_argument("--accu_step", type=int, default=8)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--pretrain", type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
