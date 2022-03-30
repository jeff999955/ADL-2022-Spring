import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import STDataset
from model import STClassifier
from utils import Vocab

from torch.utils.data import DataLoader
from tqdm import tqdm


def post_process(results, lengths, idx2tags):
    assert len(results) == len(lengths), "Length and Results length differ"
    ret= []

    for res, _len in zip(results, lengths):
        tmp = res[:_len].cpu().numpy()
        app = " ".join([idx2tags[i] for i in tmp])
        ret.append(app)
    return ret


@torch.no_grad()
def test(model, device, data_loader, idx2tags):

    model.eval()

    results, lengths, ids = [], [], []
    for tokens, length, id in tqdm(data_loader):
        tokens = tokens.to(device)
        logits = model(tokens)
        results.extend(logits.argmax(dim=-1))
        lengths.extend(length)
        ids.extend(id)

    ret = post_process(results, lengths, idx2tags)
    return ret, ids


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tags_idx_path = args.cache_dir / "tag2idx.json"
    tags2idx: Dict[str, int] = json.loads(tags_idx_path.read_text())
    idx2tags = {k: v for k, v in enumerate(tags2idx)}

    data = json.loads(args.test_file.read_text())
    dataset = STDataset(data, vocab, tags2idx, args.max_len, mode="test")
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = STClassifier(embeddings, args.hidden_size, args.num_layers, recurrent=args.recurrent,
                         dropout=args.dropout, bidirectional=args.bidirectional).to(args.device)

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print("Inferencing with {}-th epoch".format(checkpoint["epoch"]))
    print(model)
    model.eval()
    # load weights into model

    # TODO: predict dataset
    ret, ids = test(model, args.device, test_loader, idx2tags)

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        print("id,tags", file=f)
        for r, i in zip(ret, ids):
            print("{},{}".format(i, r), file=f)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path,
                        default="pred.slot.csv", required=True)

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--recurrent", type=str,
                        choices=["rnn", "lstm", "gru"], default="gru")

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
