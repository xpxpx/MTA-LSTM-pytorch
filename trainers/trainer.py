import torch
import torch.nn as nn
from utils.constant import PAD
from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, model, dataloader=None):
        super(Trainer, self).__init__(config)
        self.device = torch.device('cuda:' + str(config.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataloader = dataloader

        if config.optimizer == 'SGD':
            pass
        elif config.optimizer == 'Adam':
            pass
        else:
            raise KeyError('Unknown optimizer.')

        self.crit = nn.CrossEntropyLoss(ignore_index=config.word_vocab.get_index(PAD))

    def train(self, start_epoch=0):
        best = 0.0
        for epoch in range(start_epoch + 1, self.config.epochs):
            self.model.train()
            for step, data in enumerate(self.dataloader['train']):
                self.optimizer.zero_grad()
                pass
            pass
        pass

    def evaluate(self):
        pass
