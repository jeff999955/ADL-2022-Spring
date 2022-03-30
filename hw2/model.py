from transformers import AutoModelForMultipleChoice, AutoModelForQuestionAnswering
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleChoiceModel(nn.Module):
    def __init__(self, args, config, namae=None):
        super(MultipleChoiceModel, self).__init__()
        self.name = namae if namae is not None else args.model_name

        self.model = AutoModelForMultipleChoice.from_pretrained(
            self.name, config=config
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_bert(self):
        print("Freezing BERT")
        for param in self.model.bert.parameters():
            param.requires_grad = False


class QuestionAnsweringModel(nn.Module):
    def __init__(self, args, config, namae=None):
        super(QuestionAnsweringModel, self).__init__()
        self.name = namae if namae is not None else args.model_name
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.name, config=config
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_bert(self):
        print("Freezing BERT")
        for param in self.model.bert.parameters():
            param.requires_grad = False


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
    from transformers import AutoConfig

    args = parse_args()
    config = AutoConfig.from_pretrained(args.model_name, return_dict=False)

    model = QuestionAnsweringModel(args, config)

    print(model)
