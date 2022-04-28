# ADL HW 2

## Tasks

1. Natural Language Generation

## Environment
The required packages are listed in `requirements.txt`. 
In addition to packages allowed by TA, I used `wandb` to record the training statistics.

## How to train with my code
### Training      
Recommended:
```shell
accelerate launch train.py [OPTIONS]
```

If you wish to monitor the execution time and GPU usage, run
```shell
python3 monitor.py [command] [args]
```

Please use `python3 train.py -h` for detailed options.

## Note

The `--device` option is deprecated as the package 🤗  `accelerate` is used to manage devices. To force the accelerator use some device, please pass the environment variable `CUDA_VISIBLE_DEVICES` to the process.

### Update
The device management seems not working in current `accelerate` package, pass `CUDA_VISIBLE_DEVICES` to force `accelerate` use GPU rather than CPU by default.

## Statistics

The result of all experiments and interactive graphs can be accessed here. 
1. [News Summarization](https://wandb.ai/neverloses/News%20Summarization)
