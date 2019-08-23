
import os
import h5py
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit


problem_title = 'Dreem slow oscillation prediction'

_target_column_name = 'target'
_prediction_label_names = [0, 1, 2]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.Accuracy(name='acc')
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)



def _read_data(path, f_prefix):

    f_name = 'x_{}.h5'.format(f_prefix)
    X = h5py.File(os.path.join(path, 'data', f_name), 'r')
    X_df = pd.DataFrame(X['features'][:,:])
    X_df.columns = X.attrs['column_names']

    f_name = 'y_{}.h5'.format(f_prefix)
    y_array = h5py.File(os.path.join(path, 'data', f_name), 'r')
    y_array = y_array['target'][:].reshape(-1,)

    return X_df, y_array


def get_train_data(path='.'):
    return _read_data(path, 'train')


def get_test_data(path='.'):
    return _read_data(path, 'test')