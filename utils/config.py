import json
import pickle
import numpy as np
import jsonlines as jl
from pathlib import Path
from utils.vocab import Vocab
from utils.constant import UNK, PAD, START, END


class Config:
    def __init__(self, config_file):
        self.load_config_file(config_file)
        self.raw_config_data = json.load(open(config_file, 'r', encoding='utf-8'))

    def load_config_file(self, config_file):
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        for k, v in config.items():
            setattr(self, k, v)

    def build_vocab(self):
        self.word_vocab = Vocab('word', [UNK, PAD, START, END])
        self.pretrain_word_embedding = None

        self.build_word_vocab(self.train_file)
        self.build_word_vocab(self.dev_file)
        self.load_pretrain_word_embedding(self.embedding_file)

    def build_word_vocab(self, file):
        with jl.open(file, 'r') as f:
            for line in f:
                for word in line['sentence']:
                    self.word_vocab.add(word)

                for word in line['topic']:
                    self.word_vocab.add(word)

    def load_pretrain_word_embedding(self, embedding_file):
        scale = np.sqrt(3.0 / self.word_embedding_dim)
        pretrain_word_embedding = np.random.uniform(-scale, scale, [self.word_vocab.size(), self.word_embedding_dim])

        with jl.open(embedding_file, 'r') as f:
            for line in f:
                if line['token'] in self.word_vocab.instance2index:
                    pretrain_word_embedding[self.word_vocab.get_index(line['token'])] = line['vec']

        self.pretrain_word_embedding = pretrain_word_embedding

    def save_vocab(self, vocab_file):
        # make sure vocab dir exists
        Path('./vocab').mkdir(parents=True, exist_ok=True)
        pickle.dump({
            'word_vocab': self.word_vocab,
            'pretrain_word_embedding': self.pretrain_word_embedding
        }, open(vocab_file, 'wb'))

    def load_vocab(self, vocab_file):
        vocab_data = pickle.load(open(vocab_file, 'rb'))
        self.word_vocab = vocab_data['word_vocab']
        self.pretrain_word_embedding = vocab_data['pretrain_word_embedding']
