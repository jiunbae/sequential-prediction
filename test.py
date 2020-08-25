import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model import Model
from data import Dataset


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true),)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))
    return 100. * np.mean(diff)


def main(args: argparse.Namespace):
    dataset = Dataset(args.train)

    model_args = {
        'hidden_size': args.hidden_size,
        'input_size': args.input_size,
        'feature_size': len(dataset.dataframe.columns),
    }

    train_x, train_y = dataset(**model_args)
    test_frame = dataset.load(args.test)
    test = np.concatenate((train_x[0],
                           dataset.transform(np.hstack((test_frame, np.zeros((len(test_frame), 1)))))))

    model = Model.from_file(args.weight)

    # Test sequences
    print('Testing ...')
    for index in range(len(test) - args.input_size):
        test_input = np.expand_dims(test[index:index + args.input_size], 0)
        pred = model.predict(test_input).squeeze()
        test[index + args.input_size, -1] = pred

    test_frame[dataset.dataframe.columns[-1]] = dataset.inverse_transform(test[args.input_size:, -1])
    test_frame.to_csv(str(out.joinpath('test-prediction.csv')), index=None)

    prediction = test_frame[dataset.dataframe.columns[-1]]
    prediction += abs(prediction.min())
    label = pd.read_csv(args.label).values[:, -1]

    label = MinMaxScaler().fit_transform(label.reshape(-1, 1))
    prediction = MinMaxScaler().fit_transform(prediction.reshape(-1, 1))

    mse = mean_squared_error(label, prediction)
    mae = mean_absolute_error(label, prediction)
    mape = mean_absolute_percentage_error(label, prediction, args.epsilon)

    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scripts for data evaluation')
    parser.add_argument('--weight', required=False, default='./results/model.h5', type=str,
                        help="Model weight file")

    parser.add_argument('--train', required=True, type=str,
                        help="Input train sequential data")
    parser.add_argument('--test', required=True, type=str,
                        help="Input test sequential data")
    parser.add_argument('-l', '--label', required=False, default='./data/test-label.csv', type=str,
                        help="Input test label data file path")
    parser.add_argument('-o', '--output', required=False, default='./results', type=str,
                        help="Ouput directory")
                        
    parser.add_argument('--hidden-size', required=False, default=128, type=int,
                        help="Training arguments for network, hidden layer size")
    parser.add_argument('--input-size', required=False, default=128, type=int,
                        help="Training arguments for network, input size")

    parser.add_argument('--epsilon', required=False, default=1e-4, type=float,
                        help="Epsilon")

    main(parser.parse_args())
