export DATA_TYPE=in_domain
export ROOT_PATH=..
export DEVICE=1
export PRE_TRAINED=grf-anlg
CUDA_VISIBLE_DEVICES=${DEVICE} \
python3 main.py \
--train_data_file ${ROOT_PATH}/data/${DATA_TYPE}/train \
--dev_data_file ${ROOT_PATH}/data/${DATA_TYPE}/dev \
--test_data_file ${ROOT_PATH}/data/${DATA_TYPE}/test \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE} \
--source_length 32 \
--target_length 16 \
--model_type gpt2 \
--model_name_or_path ${ROOT_PATH}/models/${PRE_TRAINED} \
--do_eval \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--aggregate_method max \
--gamma 0.5 \
