import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from model import Model
from data import Dataset


def init(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))
    return 100. * np.mean(diff)


def main(args: argparse.Namespace):
    init(args.seed)

    dataset = Dataset(args.train)

    model_args = {
        'hidden_size': args.hidden_size,
        'input_size': args.input_size,
        'feature_size': len(dataset.dataframe.columns),
        'nested': args.nested,
        'dropout': args.dropout,
    }
    optim_args = {
        'lr': args.lr,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'decay': args.decay,
    }

    train_x, train_y = dataset(**model_args)
    test_frame = dataset.load(args.test)
    label = test_frame.values[:, -1]
    test = np.concatenate((train_x[0],
                           dataset.transform(np.hstack((test_frame.values[:, :-1], np.zeros((len(test_frame), 1)))))))

    if args.use_test:
        test_y = test_frame.values[args.input_size:, -1]
        test_x = np.empty((len(test_frame) - args.input_size, args.input_size, len(dataset.dataframe.columns)))

        for i in range(len(test_y) - args.input_size):
            test_x[i] = test_frame.values[i:i+args.input_size]

        train_x = np.vstack((train_x, test_x))
        train_y = np.concatenate((train_y, test_y))

    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)

    if args.model:
        print(f'Model load from {args.model} ...')
        model = Model.from_file(args.model)

    else:
        model = Model(model_args, optim_args)

        if not args.silence:
            model.summary()

        # Train sequences
        print('Training ...')
        model.fit(train_x, train_y, epochs=args.epoch, shuffle=False,
                  batch_size=args.batch, verbose=not args.silence,
                  callbacks=model.callbacks(early_stop=not args.no_stop))
        model.save(str(out.joinpath('model.h5')))

    # Test sequences
    print('Testing ...')
    for index in tqdm(range(len(test) - args.input_size)):
        test_input = np.expand_dims(test[index:index + args.input_size], 0)
        pred = model.predict(test_input).squeeze()
        test[index + args.input_size, -1] = pred

    test_frame[dataset.dataframe.columns[-1]] = dataset.inverse_transform(test[args.input_size:, -1])
    test_frame.to_csv(str(out.joinpath('prediction.csv')), index=None)

    if not args.no_fig:
        import matplotlib.pyplot as plt

        prediction = test_frame.values[:, -1]
        x_range = np.arange(np.size(label, 0))
        error = mean_absolute_percentage_error(label, prediction)
        plt.title(f'MAPE: {error:.4}')
        plt.ylim(0, 1200)
        plt.plot(x_range, label, c='r')
        plt.plot(x_range, prediction, c='b')
        plt.savefig('figure.jpg', dpi=400)

        print(f'MAPE: {error:.4}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=str,
                        help="Input train sequential data")
    parser.add_argument('--test', required=True, type=str,
                        help="Input test sequential data")
    parser.add_argument('-o', '--output', required=False, default='./results', type=str,
                        help="Ouput directory")

    # resume model
    parser.add_argument('--model', required=False, default='', type=str,
                        help="Load model")

    # Training arguments
    parser.add_argument('--epoch', required=False, default=100, type=int,
                        help="Training arguments for trainer, epoch")
    parser.add_argument('--batch', required=False, default=1024, type=int,
                        help="Training arguments for trainer, batch")
    parser.add_argument('--loss', required=False, default='mse', type=str,
                        choices=['mse', 'mape', 'mae'],
                        help="Training arguments for trainer, loss function")
                        
    parser.add_argument('--input-size', required=False, default=32, type=int,
                        help="Training arguments for network, input size")
    parser.add_argument('--hidden-size', required=False, default=128, type=int,
                        help="Training arguments for network, hidden layer size")
    parser.add_argument('--nested', required=False, default=2, type=int,
                        help="Training arguments for network, nested")
    parser.add_argument('--dropout', required=False, default=.1, type=float,
                        help="Training arguments for network, dropout")

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
    parser.add_argument('--no-fig', required=False, default=False, action='store_true',
                        help="If enabled, do not store figure")
    
    # Experimental only
    parser.add_argument('--use-test', required=False, default=False, action='store_true',
                        help="If enabled, use testset on train scope")
    
    main(parser.parse_args())
