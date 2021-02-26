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
        topic_embed = self.embedding_dropout(self.word_embedding(inputs['topic']))
        topic_embed = topic_embed.mean(dim=1)

        decoder_step = inputs['decoder_word'].size(1)

        total_output = []
        hidden = None
        for step in range(decoder_step):
            if step == 0:
                input_embed = topic_embed
            else:
                input_embed = self.embedding_dropout(self.word_embedding(inputs['word'][:, step]))

            hidden = self.decoder(input_embed, hidden)
            output = self.output(hidden[0])
            total_output.append(output)

        return total_output

    def predict(self, inputs, max_decoder_step=100):
        topic_embed = self.embedding_dropout(self.word_embedding(inputs['topic']))
        topic_embed = topic_embed.mean(dim=1)

        pass
