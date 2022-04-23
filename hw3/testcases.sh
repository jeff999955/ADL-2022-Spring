score () {
    python3 eval.py -r data/public.jsonl -s test.jsonl
}

python3 inference.py
score
python3 inference.py --do_sample
score
