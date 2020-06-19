import argparse
from pathlib import Path

import pandas as pd 


def main(args: argparse.Namespace):
    df = pd.read_csv(args.data)
    train, test = df.copy(), df.copy()

    train_columns = list(
        filter(len, map(str.strip, args.train_columns.split(','))))
    test_columns = list(
        filter(len, map(str.strip, args.test_columns.split(','))))
    label_columns = list(
        set(train_columns) - set(test_columns))

    for col, val in zip(['Year', 'Month', 'Day'], map(int, args.train.split('-'))):
        train = train.loc[train[col] == val]

    for col, val in zip(['Year', 'Month', 'Day'], map(int, args.test.split('-'))):
        test = test.loc[test[col] == val]

    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)

    train[train_columns].to_csv(str(out.joinpath('train.csv')), index=None)
    test[test_columns].to_csv(str(out.joinpath('test.csv')), index=None)

    test[label_columns].to_csv(str(out.joinpath('test-label.csv')), index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scripts for extract train data')
    parser.add_argument('-d', '--data', required=False, default='./data/raw.csv', type=str,
                        help="Input data file path")
    parser.add_argument('-o', '--output', required=False, default='./data', type=str,
                        help="Extracted output directory path")

    parser.add_argument('--train', required=False, default='2017', type=str,
                        help="Train target date for extraction. (YYYY(-MM(-DD)))")
    parser.add_argument('--train-columns', required=False, default='Month,Day,Hour,Quarter,P1(DayOfWeek),P2(Holiday),P3(HighestTemp),P4(Weather),Demand', type=str,
                        help="Train target columns from dataset '..., Demand' must match input size")
    parser.add_argument('--test', required=False, default='2018', type=str,
                        help="Test target date for extraction. (YYYY(-MM(-DD)))")
    parser.add_argument('--test-columns', required=False, default='Month,Day,Hour,Quarter,P1(DayOfWeek),P2(Holiday),P3(HighestTemp),P4(Weather)', type=str,
                        help="Test target columns from dataset '..., ' must match input size except y(Demand)")
    main(parser.parse_args())
