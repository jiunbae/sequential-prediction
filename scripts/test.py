import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main(args: argparse.Namespace):
    label = pd.read_csv(args.label).values[:, -1]
    prediction = pd.read_csv(args.pred).values[:, -1]
    prediction += abs(prediction.min())
    print(label.min(), label.max())
    print(prediction.min(), prediction.max())

    label = MinMaxScaler().fit_transform(label.reshape(-1, 1))
    prediction = MinMaxScaler().fit_transform(prediction.reshape(-1, 1))

    mae = np.sum(np.absolute(label - prediction))
    mse = np.square(label - prediction).mean(axis=0).item()
    print(f'MAE: {mae:.4f}, MSE: {mse:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scripts for extract train data')
    parser.add_argument('-l', '--label', required=False, default='./data/test-label.csv', type=str,
                        help="Input test label data file path")
    parser.add_argument('-p', '--pred', required=False, default='./results/prediction.csv', type=str,
                        help="Input prediction data file path")
    main(parser.parse_args())
