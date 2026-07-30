from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class TSFE(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols,  feature_config):#, target_col='label1'):
        self.feature_cols = feature_cols
        self.feature_config = feature_config
    #==========
    def fit (self, X, y=None):
        return self
    #=============
    def _fill_Nan(self, df, feature_cols):
        df[feature_cols] = df[feature_cols].ffill().fillna(df[feature_cols].median()) # df[feature_cols].median()
        return df

    def _gen_single_feature(self, df, opt, feature, period):
        if opt == 'diff':
            for p in period:
                print(p)
                df[f'{feature}_diff_{p}'] = (df[f'{feature}'].diff(p))
        elif opt == 'lag':
            for p in period:
                df[f'{feature}_lag_{p}'] = (df[f'{feature}'].shift(p))
        elif opt == 'roll_std':
            for p in period:
                df[f'{feature}_roll_std_{p}'] = (df[f'{feature}'].rolling(window=p, closed='left').std())
        elif opt == 'roll_mean_percent_res':
            for p in period:
                mean_col = df[f'{feature}'].rolling(window=p, closed='left').mean()
                df[f'{feature}_roll_mean_{p}'] =mean_col #.fillna(df[f'{feature}_roll_mean_{p}'].median()) # self not included
                df[f'{feature}_residual_{p}'] = ((df[f'{feature}']- mean_col)/(mean_col+1e-7))*100 # self not included

        elif opt =='log':
            df[f'{feature}_log'] = np.sign(df[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))
        elif opt == 'relative_per':
            for p in period:
                roll = df[f'{feature}'].rolling(window=p, closed='left')
                df[f'{feature}_relative_per_{p}'] = (df[f'{feature}']-(roll.min()))/((roll.max())-(roll.min())+1e-7)
        elif opt == 'per_rank':
            for p in period:
                df[f'{feature}_per_rank_{p}'] = df[f'{feature}'].rolling(window=p, closed='left').rank(pct=True)
    def _gen_cross_feature(self, df, opt, feat, period):
        f0, f1 = feat[0], feat[1]
        if opt == 'ratio':
            df[f'{f0}_{f1}_ratio'] = df[f0]/df[f1]
        elif opt == 'diff_cross':
            df[f'{f0}_{f1}_diff_cross'] = df[f0]-df[f1]
        elif opt == 'multi':
            df[f'{f0}_{f1}_multi'] = df[f0]*df[f1]
        elif opt == 'per_change':
            df[f'{f0}_{f1}_per_change'] = (df[f0]/df[f1]+1e-7)*100
        elif opt == 'Z_score_res':
            df[f'{feat[0]}_{feat[3]}_zcore_res_gen'] = ((df[feat[0]]-df[feat[1]])/(df[feat[2]]+1e-7))

    #=================
    def _feature_eng_apply(self, df, config):
        cross_opts = {'ratio', 'diff_cross', 'multi', 'per_change', 'Z_score_res'}
        for opt, params in config.items():
            cols = params['cols']
            period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1
            for feature in cols:
                if opt in cross_opts:
                    self._gen_cross_feature(df, opt, feature, period)
                else:
                    self._gen_single_feature(df, opt, feature, period)
        return df 

    def transform(self, X):
        df = X.copy(deep=False)
        df.index = pd.to_datetime(df.index)
        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values
        #df.index = pd.to_datetime(df.index)
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values
        df = self._fill_Nan(df, df.columns.tolist())

        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df