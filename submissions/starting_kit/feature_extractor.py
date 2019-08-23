import numpy as np
import pandas as pd

class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):

        features = pd.DataFrame(X_df.iloc[:, :11])
        features['Max'] = X_df.iloc[:,11:].max(axis = 1)
        features['Min'] = X_df.iloc[:,11:].min(axis = 1)
        features['Mean'] = X_df.iloc[:,11:].mean(axis = 1)

        return features



       