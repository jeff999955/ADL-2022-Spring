from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification


class IntentClassifier(nn.Module):
    def __init__(
        self,
        config
    ) -> None:
        super(IntentClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_config(config)

    def forward(self, labels= None, **kwargs):
        return self.model(labels= labels, **kwargs)


class STClassifier(nn.Module):
    def __init__(
        self,
        config
    ) -> None:
        super(STClassifier, self).__init__()
        self.model = AutoModelForTokenClassification.from_config(config)

    def forward(self, labels= None, **kwargs):
        return self.model(labels= labels, **kwargs)

