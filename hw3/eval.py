import json
import argparse
from tw_rouge import get_rouge


def main(args):
    refs, preds = {}, {}

    with open(args.reference) as file:
        for line in file:
            line = json.loads(line)
            refs[line["id"]] = line["title"].strip() + "\n"

    with open(args.submission) as file:
        for line in file:
            line = json.loads(line)
            preds[line["id"]] = line["title"].strip() + "\n"

    keys = refs.keys()
    refs = [refs[key] for key in keys]
    preds = [preds[key] for key in keys]

    result = get_rouge(preds, refs)
    r1, r2, rL = result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    print(f"{r1:.4f} & {r2:.4f} & {rL:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference")
    parser.add_argument("-s", "--submission")
    args = parser.parse_args()
    main(args)
