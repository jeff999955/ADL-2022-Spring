from transformers import AutoModelForMultipleChoice
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleChoiceModel(nn.Module):
    def __init__(self, args, config, out_dim = 1):
        super(MultipleChoiceModel, self).__init__()
        self.model = AutoModelForMultipleChoice.from_pretrained(args.model_name, config=config).to(args.device)
        self.model.classifier = nn.Linear(768, out_dim)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
