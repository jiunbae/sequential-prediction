from typing import Optional, List, Union, Callable, Any

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow import set_random_seed


class Model:
    def __init__(self, 
                 graph_param: Optional[dict], optim_param: Optional[dict],
                 metrics: List[Union[str, Callable]] = ['acc', 'mae', 'mape', 'mse']):
        self.model = self.graph(**graph_param)
        self.optim = self.optimizer(**optim_param)

        self.model.compile(loss='mape',
                           optimizer=self.optim, metrics=metrics)

    def __getattr__(self, key: str) -> Any:
        if hasattr(self.model, key):
            return getattr(self.model, key)
        else:
            super(Model, self).__getattribute__(key)

    @staticmethod
    def from_file(path: str):
        return keras.models.load_model(path)

    @staticmethod
    def callbacks(early_stop: bool) -> List[keras.callbacks.Callback]:
        results = []

        if early_stop:
            results += [EarlyStopping(monitor='loss', patience=20, verbose=1)]

        return results

    @staticmethod
    def graph(hidden_size: int, input_size: int, feature_size: int, nested: int = 2):
        graph = Sequential(name='Sequential Model')
        for _ in range(nested):
            graph.add(
                LSTM(hidden_size, input_shape=(input_size, feature_size), return_sequences=True))
            graph.add(
                Dropout(.1))
        graph.add(
            LSTM(hidden_size, input_shape=(input_size, feature_size)))
        graph.add(
            Dropout(.1))
        graph.add(
            Dense(1))

        return graph

    @staticmethod
    def optimizer(lr: float = .01, beta_1: float = .9, beta_2: float = .999, decay: float = .0):
        return keras.optimizers.Adam(lr=.01, beta_1=.9, beta_2=.999, decay=.0)

    @staticmethod
    def init(seed: int):
        set_random_seed(seed)
