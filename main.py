import tensorflow as tf
from keras import backend as K
import argparse
import random
from pathlib import Path

import pandas as pd
import numpy as np

from model import Model
from data import Dataset


def init(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main(args: argparse.Namespace):
    init(args.seed)

    train_set, test_set = Dataset(args.train), Dataset(args.test, test=True)
    assert len(train_set.dataframe.columns) == len(test_set.dataframe.columns) + 1,\
        "The number of columns in the train set and test set must be one difference."

    model_args = {
        'hidden_size': args.hidden_size,
        'input_size': args.input_size,
        'feature_size': len(train_set.dataframe.columns),
    }
    optim_args = {
        'lr': args.lr,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'decay': args.decay,
    }

    train_x, train_y = train_set(**model_args)
    test = np.concatenate((train_x[-1], test_set(**model_args)))
    
    model = Model(model_args, optim_args)

    if not args.silence:
        model.summary()

    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)

    # Train sequences
    model.fit(train_x, train_y, epochs=args.epoch,
              batch_size=args.batch, verbose=not args.silence, 
              callbacks=model.callbacks(early_stop=not args.no_stop))
    model.save(str(out.joinpath('model.h5')))

    # Test sequences
    for index in range(len(test_set)):
        test_input = np.expand_dims(test[index:index + args.input_size], 0)
        pred = model.predict(test_input).squeeze()
        test[index + args.input_size, -1] = pred

    result = test_set.dataframe
    # result[train_set.dataframe.columns[-1]] = train_set.inverse_transform(test[args.input_size:, -1])
    result[train_set.dataframe.columns[-1]] = test[args.input_size:, -1]
    result.to_csv(str(out.joinpath('prediction.csv')), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=str,
                        help="Input train sequential data")
    parser.add_argument('--test', required=True, type=str,
                        help="Input test sequential data")
    parser.add_argument('-o', '--output', required=False, default='./results', type=str,
                        help="Ouput directory")

    # Training arguments
    parser.add_argument('--epoch', required=False, default=100, type=int,
                        help="Training arguments for trainer, epoch")
    parser.add_argument('--batch', required=False, default=32, type=int,
                        help="Training arguments for trainer, batch")
    parser.add_argument('--loss', required=False, default='mse', type=str, choices=['mse', 'mae'],
                        help="Training arguments for trainer, loss function")
                        
    parser.add_argument('--hidden-size', required=False, default=32, type=int,
                        help="Training arguments for network, hidden layer size")
    parser.add_argument('--input-size', required=False, default=8, type=int,
                        help="Training arguments for network, input size")

    parser.add_argument('--no-stop', required=False, default=False, action='store_true',
                        help="If enabled, prevent early stopping")
    parser.add_argument('--lr', required=False, default=.01, type=float,
                        help="Training arguments for optimizer, learing rate")
    parser.add_argument('--beta-1', required=False, default=.9, type=float,
                        help="Training arguments for optimizer, beta 1")
    parser.add_argument('--beta-2', required=False, default=.999, type=float,
                        help="Training arguments for optimizer, beta 2")
    parser.add_argument('--decay', required=False, default=.0, type=float,
                        help="Training arguments for optimizer, decay")

    # ETC
    parser.add_argument('--seed', required=False, default=42, type=int,
                        help="The answer to life the universe and everything")
    parser.add_argument('--silence', required=False, default=False, action='store_true',
                        help="Verbose log")
    
    main(parser.parse_args())
