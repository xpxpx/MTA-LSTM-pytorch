import torch
import torch.nn as nn
from layers.embedding import TokenEmbedding
from layers.decoder import LSTMDecoder
from utils.constant import PAD


class TopicAveragedLSTM(nn.Module):
    def __init__(self, config):
        super(TopicAveragedLSTM, self).__init__()
        self.word_embedding = TokenEmbedding(
            config.word_vocab.size(),
            config.word_embedding_dim,
            padding_idx=config.word_vocab.get_index(PAD),
            pretrain_embedding=config.pretrain_word_embedding,
            parameter_update=config.word_embedding_update
        )
        self.encoder = nn.Linear(
            config.word_embedding_dim,
            config.decoder_hidden_dim
        )
        self.decoder = LSTMDecoder(
            config.word_embedding_dim,
            config.decoder_hidden_dim
        )
        self.output = nn.Linear(
            config.decoder_hidden_dim,
            config.word_vocab.size()
        )

        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.encoder_dropout = nn.Dropout(config.encoder_dropout)
        self.decoder_dropout = nn.Dropout(config.decoder_dropout)

    def forward(self, inputs):
        topic_embed = self.embedding_dropout(self.word_embedding(inputs['encoder_topic']))

        max_decoder_step = inputs['decoder_word'].size(1)

        total_output = []
        for step in range(1, max_decoder_step):
            hidden = self.decoder(inputs['decoder_word'][:, step])
            output = self.output(hidden[0])
            total_output.append(output)

        return total_output

    def predict(self, inputs):
        pass
