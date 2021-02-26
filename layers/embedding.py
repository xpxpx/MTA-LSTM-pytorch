import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=None, pretrain_embedding=None, parameter_update=True):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrain_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrain_embedding))

        if not parameter_update:
            self.embedding.weight.requires_grad_(parameter_update)

    def forward(self, token):
        return self.embedding(token)
