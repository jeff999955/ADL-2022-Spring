export CUDA_VISIBLE_DEVICES=0
python3 inference.py --num_beams 5 --batch_size 32 --test_file $1 --out_json $2
unset CUDA_VISIBLE_DEVICES