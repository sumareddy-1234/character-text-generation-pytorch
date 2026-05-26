import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x, hidden):

        x = self.embedding(x)

        out, hidden = self.lstm(x, hidden)

        out = out.contiguous().view(
            -1,
            self.hidden_dim
        )

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        hidden = (
            weight.new(
                self.n_layers,
                batch_size,
                self.hidden_dim
            ).zero_(),

            weight.new(
                self.n_layers,
                batch_size,
                self.hidden_dim
            ).zero_()
        )

        return hidden