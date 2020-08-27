import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))
    return 100. * np.mean(diff)


def scale(values):
    min_ = values.min()
    max_ = values.max()
    
    return (values - min_) / (max_ - min_)


def main(args: argparse.Namespace):
    try:
        label = pd.read_csv(args.label).values[:, -1]
        pred_data = pd.read_csv(args.pred).values
        pred = pred_data[:, -1]
    except Exception as e:
        try:
            # legacy
            label = pd.read_csv(args.label).values[:, -1]
            pred_data = pd.read_csv(args.pred).values
            pred = pred_data[1:, -1]
        except Exception as e:
            print (e)

    dest = Path(args.output)
    dest.mkdir(exist_ok=True, parents=True)

    if args.scale:
        label = (label - label.min()) / (label.max() - label.min())
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    
    d = dest.joinpath('days')
    d.mkdir(exist_ok=True)
    days = pred_data[:, :2]
    days_unique = np.sort(np.unique(days, axis=0))
    mape_days = np.hstack((days_unique, np.empty((len(days_unique), 1))))
    for i, day in enumerate(days_unique):
        index = np.where((days == day).all(axis=1))
        mo, da = day
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        error = mean_absolute_percentage_error(label[index], pred[index])
        ax.set_title(f'MAPE: {error:.4}')
        mape_days[i, -1] = error
        ax.set_ylim((0, 1 if args.scale else 1200))
        ax.set_xlim((0, 96))
        ax.plot(label[index], c='r')
        ax.plot(pred[index], c='b')
        fig.savefig(str(d.joinpath(f'{int(mo):02}-{int(da):02}.jpg')), dpi=400)

    d = dest.joinpath('months')
    d.mkdir(exist_ok=True)
    months = pred_data[:, :1]
    months_unique = np.unique(months, axis=0)
    for month in months_unique:
        index = np.where((months == month).all(axis=1))
        mo = month
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'MAPE: {mean_absolute_percentage_error(label[index], pred[index]):.4}')
        ax.set_ylim((0, 1 if args.scale else 1200))
        ax.set_xlim((0, index[0].size))
        ax.plot(label[index], c='r')
        ax.plot(pred[index], c='b')
        fig.savefig(str(d.joinpath(f'{int(mo):02}.jpg')), dpi=400)
    
    pd.DataFrame(mape_days, columns=['Month', 'Day', 'MAPE']).to_csv(str(dest.joinpath('mape-per-days.csv')), index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scripts for visualization')
    parser.add_argument('-l', '--label', required=False, default='./data/test-label.csv', type=str,
                        help="Input test label data file path")
    parser.add_argument('-p', '--pred', required=False, default='./results/prediction.csv', type=str,
                        help="Input prediction data file path")
    parser.add_argument('--output', required=False, default='./figures', type=str,
                        help="Output directory")

    parser.add_argument('--scale', required=False, action='store_true', default=False,
                        help="scale 0..1")

    main(parser.parse_args())
