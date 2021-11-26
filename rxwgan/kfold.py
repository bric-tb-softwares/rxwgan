
__all__ = ['KFold']

import numpy as np
from sklearn.model_selection import KFold as KFold_skl

class KFold:
    def __init__(self, n_splits=2, random_state=None, shuffle=False):
        self.kf = KFold_skl(n_splits = n_splits, shuffle = shuffle, random_state = random_state)

    def get_n_splits(self):
        return self.kf.get_n_splits()

    def split(self, dataframe):
        splits = list((train_index, val_index) for train_index, val_index in self.kf.split(dataframe))
        return [(self.__remove_items(splits[idx][0], splits[(idx+1)%self.get_n_splits()][-1]), splits[idx][-1], splits[(idx+1)%self.get_n_splits()][-1]) for idx, item in enumerate(splits)]

    def __make_set_array(self, array):
        return np.array(list(array))
    
    def __remove_items(self, array1, array2):
        return self.__make_set_array(set(array1).difference(set(array2)))

