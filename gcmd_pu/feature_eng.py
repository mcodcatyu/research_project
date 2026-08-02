from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import numpy as np


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

    def _gen_single_feature(self, df, new_cols, opt, feature, period):
        if opt == 'diff':
            for p in period:
                new_cols[f'{feature}_diff_{p}'] = (new_cols[f'{feature}'].diff(p))
        elif opt == 'lag':
            for p in period:
                new_cols[f'{feature}_lag_{p}'] = (new_cols[f'{feature}'].shift(p))
        elif opt == 'roll_std':
            for p in period:
                new_cols[f'{feature}_roll_std_{p}'] = (new_cols[f'{feature}'].rolling(window=p, closed='left').std())
        elif opt == 'roll_mean_percent_res':
            for p in period:
                mean_col = df[f'{feature}'].rolling(window=p, closed='left').mean()
                new_cols[f'{feature}_roll_mean_{p}'] =mean_col #.fillna(df[f'{feature}_roll_mean_{p}'].median()) # self not included
                new_cols[f'{feature}_residual_{p}'] = ((new_cols[f'{feature}']- mean_col)/(mean_col+1e-7))*100 # self not included

        elif opt =='log':
            new_cols[f'{feature}_log'] = np.sign(new_cols[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))
        elif opt == 'relative_per':
            for p in period:
                roll = new_cols[f'{feature}'].rolling(window=p, closed='left')
                new_cols[f'{feature}_relative_per_{p}'] = (new_cols[f'{feature}']-(roll.min()))/((roll.max())-(roll.min())+1e-7)
        elif opt == 'per_rank':
            for p in period:
                new_cols[f'{feature}_per_rank_{p}'] = new_cols[f'{feature}'].rolling(window=p, closed='left').rank(pct=True)



    def _gen_cross_feature(self, df, new_cols, opt, feat, period):
        f0, f1 = feat[0], feat[1]
        def get_col(col_name):
            return(
                new_cols[col_name] if col_name in new_cols else df[col_name]
            )
        if opt == 'ratio':
            new_cols[f'{f0}_{f1}_ratio'] = get_col[f0]/get_col[f1]
        elif opt == 'diff_cross':
            new_cols[f'{f0}_{f1}_diff_cross'] = get_col[f0]-get_col[f1]
        elif opt == 'multi':
            new_cols[f'{f0}_{f1}_multi'] = get_col[f0]*get_col[f1]
        elif opt == 'per_change':
            new_cols[f'{f0}_{f1}_per_change'] = (get_col[f0]/get_col[f1]+1e-7)*100
        elif opt == 'Z_score_res':
            new_cols[f'{feat[0]}_{feat[3]}_zcore_res_gen'] = ((get_col[feat[0]]-get_col[feat[1]])/(get_col[feat[2]]+1e-7))

    #=================
    def _feature_eng_apply(self, df, config):
        cross_opts = {'ratio', 'diff_cross', 'multi', 'per_change', 'Z_score_res'}
        new_cols={}
        for opt, params in config.items():
            if opt not in cross_opts:
                cols = params['cols']
                period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1
                for feature in cols:
                    self._gen_single_feature(df, opt, new_cols, feature, period)

        for opt, params in config.items():
            if opt in cross_opts:
                cols = params['cols']
                period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1
                for feature in cols:
                    self._gen_cross_feature(df, opt, new_cols, feature, period)

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)

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