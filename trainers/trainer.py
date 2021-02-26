from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, model, dataloader=None):
        super(Trainer, self).__init__(config)

    def train(self):
        pass

    def evaluate(self):
        pass
