import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BHKSizeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.bhk_size_medians_ = X.groupby('BHK')['Size'].median()
        self.bhk_fallback_ = X['BHK'].median()
        self.size_fallback_ = X['Size'].median()
        return self

    def transform(self, X):
        X = X.copy()

        def impute_bhk(row):
            if pd.notna(row['BHK']):
                return row['BHK']
            if pd.notna(row['Size']):
                diffs = (self.bhk_size_medians_ - row['Size']).abs()
                return diffs.idxmin()
            return self.bhk_fallback_

        def impute_size(row):
            if pd.notna(row['Size']):
                return row['Size']
            if pd.notna(row['BHK']) and row['BHK'] in self.bhk_size_medians_.index:
                candidate = self.bhk_size_medians_[row['BHK']]
                if pd.notna(candidate):
                    return candidate
            return self.size_fallback_

        X['BHK'] = X.apply(impute_bhk, axis=1)
        X['Size'] = X.apply(impute_size, axis=1)
        return X


class AreaLocalityTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def fit(self, X, y):
        df = X[['Area Locality']].copy()
        df['target'] = y.values

        self.global_mean_ = df['target'].mean()

        stats = df.groupby('Area Locality')['target'].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean_) / (stats['count'] + self.smoothing)

        self.locality_means_ = smoothed
        return self

    def transform(self, X):
        X = X.copy()
        X['Area Locality'] = X['Area Locality'].map(self.locality_means_).fillna(self.global_mean_)
        return X