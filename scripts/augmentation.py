import argparse
from pathlib import Path

import numpy as np
import pandas as pd 


def main(args: argparse.Namespace):
    df = pd.read_csv(args.train)
    train = df.values
    data = train.copy()
    for _ in range(args.repeat):
        dump = train.copy()
        dump[:, -1] += np.random.normal(.0, train[:, -1].std() * .1, size=train[:, -1].shape)
        data = np.concatenate((data, dump))
    pd.DataFrame(data, columns=df.columns).to_csv(args.output, index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scripts for extract train data')
    parser.add_argument('-t', '--train', required=False, default='./data/train.csv', type=str,
                        help="Input data file path")
    parser.add_argument('-o', '--output', required=False, default='./data/train-augmented.csv', type=str,
                        help="Extracted output directory path")

    parser.add_argument('--repeat', required=False, default=8, type=int,
                        help="repeat augmentation variable")

    main(parser.parse_args())
