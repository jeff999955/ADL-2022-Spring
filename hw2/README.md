# ADL HW 2

## Tasks

1. Context Selection
2. Question Answering

## Environment
The required packages are listed in `requirements.txt`. 
In addition to packages allowed by TA, I used `wandb` to record the training statistics.

## Information
This pipeline consists of two tasks, context selection and question answering, which could be trained simultaneously.

## Context Selection
### Training      
Recommended:
```shell
accelerate launch multiple_choice.py [OPTIONS]
```

Please use `python3 multiple_choice.py -h` for detailed options.

## Question Answering
### Training 
```shell
accelerate launch question_answering.py [OPTIONS]
```

Please use `python3 question_answering.py -h` for detailed options.

## Note

The `--device` option is deprecated as the package 🤗  `accelerate` is used to manage devices. To force the accelerator use some device, please pass the environment variable `CUDA_VISIBLE_DEVICES` to the process.

## Statistics

The result of all experiments and interactive graphs can be accessed here. 
1. [Context Selection](https://wandb.ai/neverloses/Context-Selection/overview?workspace=user-neverloses)
2. [Question Answering](https://wandb.ai/neverloses/Question%20Answering?workspace=user-neverloses)

## Specification
For detailed spec, please refer to `spec.pdf`.
