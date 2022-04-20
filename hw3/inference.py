
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import numpy as np

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler
from utils import same_seeds
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb

from tw_rouge import get_rouge
from utils import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5920)
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./data/public.jsonl",
    )
    
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    parser.add_argument(
        "--out_json",
        type=Path,
        default="./test.jsonl"
    )

    # data
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--max_answer_len", type=int, default = 64)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--strategy", type=str, choices=STRAT, default = STRAT[0])

    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default = 1.0)


    args = parser.parse_args()
    return args

def main(args):
    print(args)
    config = get_config(args)
    print(config)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)