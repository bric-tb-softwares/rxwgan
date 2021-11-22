import numpy as np
from sklearn.model_selection import KFold

class newKfold:
    def __init__(self, dataframe, n_splits, seed):
        self.kf = KFold(n_splits = n_splits, shuffle = True, random_state = seed)
        self.splits = list((train_index, val_index) for train_index, val_index in self.kf.split(dataframe))

    def make_set_array(self, array):
        return np.array(list(array))
    
    def remove_items(self, array1, array2):
        return self.make_set_array(set(array1).difference(set(array2)))
    
    def create_new_kfold(self):
        new_kfold = [(self.remove_items(self.splits[idx][0], self.splits[(idx+1)%self.kf.get_n_splits()][-1]), self.splits[idx][-1], self.splits[(idx+1)%self.kf.get_n_splits()][-1]) for idx, item in enumerate(self.splits)]
        return new_kfold
