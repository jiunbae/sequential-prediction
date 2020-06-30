import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true),)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-6):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))
    return 100. * np.mean(diff)


def main(args: argparse.Namespace):
    epsilon = 1e-10
    label = pd.read_csv(args.label).values[:, -1]
    prediction = pd.read_csv(args.pred).values[:, -1]
    prediction += abs(prediction.min())

    label = MinMaxScaler().fit_transform(label.reshape(-1, 1))
    prediction = MinMaxScaler().fit_transform(prediction.reshape(-1, 1))

    mse = mean_squared_error(label, prediction)
    mae = mean_absolute_error(label, prediction)
    mape = mean_absolute_percentage_error(label, prediction)

    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scripts for extract train data')
    parser.add_argument('-l', '--label', required=False, default='./data/test-label.csv', type=str,
                        help="Input test label data file path")
    parser.add_argument('-p', '--pred', required=False, default='./results/prediction.csv', type=str,
                        help="Input prediction data file path")
    main(parser.parse_args())
