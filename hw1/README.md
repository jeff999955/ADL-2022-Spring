# ADL HW 1

## Tasks

1. Intent Classification
2. Slot Tagging

## Environment
I use the environment mentioned in the spec without any addition packages.

## Preprocessing
```shell
bash preprocess.sh
```

## Intent detection
### Training      
```shell
python3 train_intent.py [OPTIONS]
```

Please use `python3 train_intent.py -h` for detailed options.

## Slot Tagging
### Training 
```shell
python3 train_slot.py [OPTIONS]
```

Please use `python3 train_slot.py -h` for detailed options.

## Statistics

The result of all experiments can be accessed here. 
1. [Intent Classification](https://wandb.ai/neverloses/intent%20classification)
2. [Intent Classification Report](https://wandb.ai/neverloses/intent%20classification/reports/Intent-Classification--VmlldzoxNjA3Nzcw)
3. [Slot Tagging](https://wandb.ai/neverloses/slot%20tagging)
4. [Slot Tagging Report](https://wandb.ai/neverloses/slot%20tagging/reports/Slot-Tagging--VmlldzoxNjA4NTMz)

## Specification
For detailed spec, please refer to `spec.pdf`.
