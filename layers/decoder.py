import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMDecoder, self).__init__()
        self.decoder = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, inputs, hidden):
        hidden = self.decoder(inputs, hidden)
        return hidden
