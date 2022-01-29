
__all__ = ['stratified_train_val_test_splits']

from sklearn.model_selection import StratifiedKFold


#
# Split train/val/test splits
#
def stratified_train_val_test_splits( df, seed=512 ):

    cv_train_test = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_train_val  = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    sorts_train_test = []

    for train_val_idx, test_idx in cv_train_test.split( df.values,df.target.values ):
        train_val_df = df.iloc[train_val_idx]
        sorts = []
        for train_idx, val_idx in cv_train_val.split( train_val_df.values, train_val_df.target.values ):
            sorts.append((train_val_df.index[train_idx].values, train_val_df.index[val_idx].values, test_idx))
        sorts_train_test.append(sorts)
    return sorts_train_test

