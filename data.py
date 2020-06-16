from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, path: str, test: bool = False):
        self.dataframe = pd.read_csv(path)
        self.is_test = test

    def __len__(self):
        return np.size(self.dataframe, 0)

    def __call__(self, hidden_size: int, input_size: int, feature_size: int) ->\
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert feature_size == len(self.dataframe.columns) + int(self.is_test),\
            "feautre_size must match length of columns"

        scaler = MinMaxScaler()
        dataframe = scaler.fit_transform(self.dataframe)

        if self.is_test:
            return np.hstack((dataframe, np.zeros((len(self), 1))))

        else:
            y = dataframe[input_size:, -1]
            x = np.empty((len(self) - input_size, input_size, feature_size))
            for i in range(len(self) - input_size):
                x[i] = dataframe[i:i+input_size]

            return x, y
