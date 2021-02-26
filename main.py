import argparse
import utils
import dataloaders
import models
import trainers
from utils.function import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='Config', help='')
    parser.add_argument('--config_file', type=str, help='')
    parser.add_argument('--save_vocab', type=str, help='')
    parser.add_argument('--load_vocab', type=str, help='')
    parser.add_argument('--vocab_file', type=str, help='')
    parser.add_argument('--run', choices=['train', 'evaluate', 'predict'], help='')
    parser.add_argument('--seed', type=int, default=123, help='')
    parser.add_argument('--gpu', type=int, default=0, help='')
    return parser.parse_args()


def train(config):
    # set random seed
    set_random_seed(config.seed)

    # build dataloader
    train_dataloader = getattr(dataloaders, config.dataloader)(config, config.train_file, config.batch_size, shuffle=True)
    dev_dataloader = getattr(dataloaders, config.dataloader)(config, config.dev_file, config.batch_size, shuffle=False)

    # build model
    model = getattr(models, config.model)(config)

    # build trainer
    trainer = getattr(trainers, config.trainer)(config, model, dataloader={'train': train_dataloader, 'dev': dev_dataloader})
    trainer.train()


def evaluate():
    pass


def predict():
    pass


def main():
    pass


if __name__ == '__main__':
    args = vars(parse_args())
    if args['run'] == 'train':
        config = getattr(utils, args['config'])(args['config_file'])

        config.gpu = args['gpu']
        config.seed = args['seed']
        config.alias += '_' + str(args['seed'])

        if args['load_vocab']:
            config.load_vocab(args['vocab_file'])
        else:
            config.build_vocab()

            # save vocab
            if args['save_vocab']:
                config.save_vocab(args['vocab_file'])

        train(config)
    elif args['run'] == 'evaluate':
        pass
    else:
        raise KeyError('Unknown run option.')
