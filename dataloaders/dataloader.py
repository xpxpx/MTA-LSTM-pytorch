import torch
import random
import jsonlines as jl
from utils.constant import PAD, START, END


class DataLoader:
    def __init__(self, config, file, batch_size, shuffle=False):
        self.file = file
        self.word_vocab = config.word_vocab

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data = []
        self.index = 0
        with jl.open(file, 'r') as f:
            for line in f:
                self.data.append(self.build(line))

        if self.shuffle:
            random.shuffle(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            if self.shuffle:
                random.shuffle(self.data)
            self.index = 0
            raise StopIteration
        else:
            current_data = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size

            # padding
            max_length = max([one['length'] for one in current_data])

            word = []
            length = []
            topic = []

            for one in current_data:
                word.append(one['word'] + [self.word_vocab.get_index(PAD)] * (max_length - one['length']))
                length.append(one['length'])
                topic.append(one['topic'])

            return {
                'word': torch.tensor(word),
                'length': torch.tensor(length),
                'topic': torch.tensor(topic)
            }

    def __len__(self):
        return len(self.data) // self.batch_size

    def build(self, line):
        topic = line['topic']
        sentence = [START] + line['sentence'] + [END]

        return {
            'word': [self.word_vocab.get_index(word) for word in sentence],
            'length': len(sentence),
            'topic': [self.word_vocab.get_index(word) for word in topic]
        }
