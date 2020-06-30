from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, path: str):
        self.dataframe = pd.read_csv(path)
        self.scaler = MinMaxScaler()
        self._min, self._max, self._scale = 0., 1., 1500

    def __len__(self):
        return np.size(self.dataframe, 0)

    def __call__(self, hidden_size: int, input_size: int, feature_size: int) ->\
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        dataframe = self.scaler.fit_transform(self.dataframe)

        y = dataframe[input_size:, -1]
        x = np.empty((len(self) - input_size, input_size, feature_size))

        for i in range(len(self) - input_size):
            x[i] = dataframe[i:i+input_size]

        return x, y

    @staticmethod
    def load(path: str):
        return pd.read_csv(path)

    def transform(self, X: np.ndarray) ->\
            np.ndarray:
        return self.scaler.transform(X)

    def inverse_transform(self, X: np.ndarray, axis: int = -1) ->\
            np.ndarray:
        # X *= self._scale
        # _min, _max = X.min(), X.max()
        # X = (X - _min) / (_max - _min) *(self._max - self._min) + self._min
        # return X
        return (X + self.scaler.min_[axis]) / self.scaler.scale_[axis]
