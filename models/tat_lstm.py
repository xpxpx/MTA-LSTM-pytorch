import torch
import torch.nn as nn
from layers.embedding import TokenEmbedding
from layers.decoder import LSTMDecoder
from layers.attention import Attention


class TopicAttentionLSTM(nn.Module):
    def __init__(self, config):
        super(TopicAttentionLSTM, self).__init__()

    def forward(self):
        pass
