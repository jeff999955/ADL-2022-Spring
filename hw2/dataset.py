import os
import json
import torch
import random
from torch.utils.data import Dataset

from tqdm import tqdm


class MultipleChoiceDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
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
                    if mode != "test":
                        label = data["paragraphs"].index(data["relevant"])
                    else:
                        label = random.choice(list(range(len(data["paragraphs"]))))

                    qa_pair = [
                        "{} {}".format(data["question"], context_data[i])
                        for i in data["paragraphs"]
                    ]
                    features = tokenizer(
                        qa_pair,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    self.json_data.append(
                        {
                            "id": data["id"],
                            "input_ids": features["input_ids"],
                            "token_type_ids": features["token_type_ids"],
                            "attention_mask": features["attention_mask"],
                            "label": label,
                        }
                    )
            os.makedirs(args.cache_dir, exist_ok=True)
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
        input_ids = torch.stack(input_ids).to(self.device)
        attention_masks = torch.stack(attention_masks).to(self.device)
        token_type_ids = torch.stack(token_type_ids).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        return ids, input_ids, attention_masks, token_type_ids, labels


class QuestionAnsweringDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", relevant=None):
        assert not (mode == "test" and relevant is None)
        self.mode = mode
        self.device = args.device
        self.tokenizer = tokenizer
        self.json_data = []
        with open(os.path.join(args.data_dir, "context.json"), "r") as f:
            self.context_data = json.load(f)
        with open(os.path.join(args.data_dir, f"{mode}.json"), "r") as f:
            json_data = json.load(f)
            print("Preprocessing QA Data:")
            for data in tqdm(json_data):
                if mode != "test":
                    label = data["relevant"]
                else:
                    label = data["paragraph"][relevant[data["id"]]]

                self.json_data.append(
                    {
                        "id": data["id"],
                        "question": data["question"],
                        "context": label,
                        "answer": data["answer"],
                    }
                )

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        return self.json_data[idx]

    def collate_fn(self, batch):
        ids = [ sample["id"] for sample in batch ]
        inputs = self.tokenizer(
            [data["question"] for data in batch],
            [self.context_data[data["context"]] for data in batch],
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["context"] = [self.context_data[data["context"]] for data in batch]
        print(inputs.keys())

        if self.mode != "test":
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                start_char = batch[sample_idx]["answer"]["start"]
                end_char = batch[sample_idx]["answer"]["start"] + len(
                    batch[sample_idx]["answer"]["text"]
                )
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if (
                    offset[context_start][0] > start_char
                    or offset[context_end][1] < end_char
                ):
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = torch.tensor(start_positions)
            inputs["end_positions"] = torch.tensor(end_positions)
        else:
            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(batch[sample_idx]["id"])

                sequence_ids = inputs.sequence_ids(i)
                offset = inputs[i]["offset_mapping"]
                inputs[i]["offset_mapping"] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                ]

            inputs["example_id"] = example_ids
        return ids, inputs


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

    parser.add_argument("--accu_step", type=int, default=8)

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

    train_set = QuestionAnsweringDataset(args, tokenizer)
    valid_set = QuestionAnsweringDataset(args, tokenizer, mode="valid")

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
    for id, data in tqdm(train_loader):
        print("train")
        for k, v in data.items():
            print(k, v.type())
        break

    for id, data in tqdm(valid_loader):
        print("valid")
        for k, v in data.items():
            print(f"{k} = inputs[\"{k}\"].to(args.device)")
        break
