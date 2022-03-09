from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        recurrent: str = "rnn",
        dropout: float = 0.5,
        bidirectional: bool = False,
        freeze: bool = False,
        num_class: int = 150,
        prefix="",
    ) -> None:
        super(IntentClassifier, self).__init__()
        self.name = f"{prefix}{recurrent}_{2 if bidirectional else 1}_{hidden_size}_{num_layers}"
        self.setting = locals()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=freeze)
        implementation = {
            "rnn": nn.RNN(300, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional),
            "lstm": nn.LSTM(300, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional),
            "gru": nn.GRU(300, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        }
        self.recurrent = implementation[recurrent]
        D = hidden_size * (2 if bidirectional else 1)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(D),
            nn.LeakyReLU(),
            nn.Linear(D, D),
            nn.Dropout(dropout),
            nn.LayerNorm(D),
            nn.LeakyReLU(),
            nn.Linear(D, num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, x, **kwargs) -> Dict[str, torch.Tensor]:
        """ 
        Input:
            x: tensor of size (batch, seq)
        Output:
            x: tensor of size (batch, num_classes)
        """
        # TODO: implement model forward

        x = self.embedding(x)
        x, h = self.recurrent(x, **kwargs)  # (batch, seq, bi * hidden)
        x = torch.sum(x, dim=1)
        x = self.mlp(x)
        return x


class STClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        recurrent: str = "rnn",
        dropout: float = 0.5,
        bidirectional: bool = False,
        freeze: bool = False,
        num_class: int = 10,  # gotta include padding
        prefix="",
    ) -> None:
        super(STClassifier, self).__init__()
        self.name = f"{prefix}{recurrent}_{2 if bidirectional else 1}_{hidden_size}_{num_layers}_slot"
        self.setting = locals()

        self.embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=freeze)
        implementation = {
            "rnn": nn.RNN(300, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional),
            "lstm": nn.LSTM(300, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional),
            "gru": nn.GRU(300, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        }
        self.prenet = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(300, 300),
            nn.ReLU()
        )
        self.recurrent = implementation[recurrent]

        D = hidden_size * (2 if bidirectional else 1)

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(D),
            nn.RReLU(),
            nn.Linear(D, D),
            nn.Dropout(dropout),
            nn.LayerNorm(D),
            nn.RReLU(),
            nn.Linear(D, num_class)
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, x, **kwargs) -> Dict[str, torch.Tensor]:
        """ 
        Input:
            x: tensor of size (batch, seq)
        Output:
            x: tensor of size (batch, num_classes)
        """
        # TODO: implement model forward

        x = self.embedding(x)
        x = self.prenet(x)
        x, h = self.recurrent(x, **kwargs)  # (batch, seq, bi * hidden)
        # print(x.shape)
        x = self.mlp(x)
        return x
