import torch
import torch.nn as nn
from layers.embedding import TokenEmbedding
from layers.decoder import LSTMDecoder
from layers.attention import Attention


class MultiTopicAwareLSTM(nn.Module):
    def __init__(self, config):
        super(MultiTopicAwareLSTM, self).__init__()
        self.word_embedding = TokenEmbedding()
        self.decoder = LSTMDecoder()

    def forward(self, inputs):
        pass
