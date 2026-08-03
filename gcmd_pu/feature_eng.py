from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import numpy as np


class TSFE(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols,  feature_config, type_col='type'):#, target_col='label1'):
        self.feature_cols = feature_cols
        self.feature_config = feature_config
        self.type_col = type_col
    #==========
    def fit (self, X, y=None):
        return self
    #=============
    def _fill_Nan(self, df, feature_cols):
        df[feature_cols] = df[feature_cols].ffill().fillna(df[feature_cols].median()) # df[feature_cols].median()
        return df

    def _gen_single_feature(self, df, opt, feature, period=None):
        if self.type_col in df.columns:
            g = df.groupby(self.type_col)[feature]
        else:
            g = df[feature]


        def _clean(reset):
            if isinstance(reset.index, pd.MultiIndex):
                return reset.reset_index(0, drop=True).sort_index()
            return reset

        
        if opt == 'diff':
            for p in period:
                df[f'{feature}_diff_{p}'] = _clean(g.diff(p))

        elif opt == 'lag':
            for p in period:
                df[f'{feature}_lag_{p}'] = _clean(g.shift(p))
        elif opt == 'roll_std':
            for p in period:
                df[f'{feature}_roll_std_{p}'] = _clean(g.rolling(window=p, closed='left').std())
        elif opt == 'roll_mean_percent_res':
            for p in period:
                mean_col = _clean(g.rolling(window=p, closed='left').mean())
                df[f'{feature}_roll_mean_{p}'] =mean_col #.fillna(df[f'{feature}_roll_mean_{p}'].median()) # self not included
                df[f'{feature}_residual_{p}'] = ((df[f'{feature}']- mean_col)/(mean_col+1e-7))*100 # self not included

        elif opt =='log':
            df[f'{feature}_log'] = np.sign(df[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))
        elif opt == 'relative_per':
            for p in period:
                roll_min = _clean(g.rolling(window=p, closed='left').min())
                roll_max = _clean(g.rolling(window=p, closed='left').max())
                df[f'{feature}_relative_per_{p}'] = (df[f'{feature}']-(roll_min))/((roll_max)-(roll_min)+1e-7)
        elif opt == 'per_rank':
            for p in period:
                df[f'{feature}_per_rank_{p}'] = _clean(g.rolling(window=p, closed='left').rank(pct=True))
        elif opt == 'roll_median':
            for p in period:
                df[f'{feature}_roll_median_{p}'] = (
                    _clean(g.rolling(window=p, closed='left').median())
                )
        elif opt == 'roll_mad':
            def _calc_mad(x):
                med = np.median(x)
                return np.median(np.abs(x-med))

            for p in period:
                df[f'{feature}_mad_{p}']=(
                    _clean(g.rolling(window=p, closed='left').apply(_calc_mad, raw=True))
                )
                
        elif opt == 'roll_median_percent_res':
            """
            強健殘差(當前點偏離中位數百分比)
            """
            for p in period:
                median_col =df[f'{feature}_roll_median_{p}']
                df[f'{feature}_robust_residual_{p}'] = (
                    (df[f'{feature}']-median_col)/(median_col +1e-7 )*100
                )

    def _gen_cross_feature(self, df, opt, feat, period):
        f0, f1 = feat[0], feat[1]
        if opt == 'ratio':
            df[f'{f0}_{f1}_ratio'] = df[f0]/df[f1]
        elif opt == 'diff_cross':
            df[f'{f0}_{f1}_diff_cross'] = df[f0]-df[f1]
        elif opt == 'multi':
            df[f'{f0}_{f1}_multi'] = df[f0]*df[f1]
        elif opt == 'per_change':
            df[f'{f0}_{f1}_per_change'] = (df[f0]-df[f1]/(df[f1]+1e-7))*100
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
        df = X.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values
        #df.index = pd.to_datetime(df.index)
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values
        df = self._fill_Nan(df, df.columns.tolist())

        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df