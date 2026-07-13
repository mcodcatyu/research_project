import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class TSFE(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols,  feature_config, target_col='label1'):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.feature_config = feature_config
    #==========
    def fit (self, X, y=None):
        return self
    #=============
    def _fill_Nan(self, df, feature_col):
        for i in feature_col:
            df[i] = df[i].ffill().fillna(0)
        return df
    # ============
    def _ratio_featutre_gen(self, df):
        df['CH4_area_ht_ratio'] = df['CH4_area']/df['CH4_ht']
        df['CH4_w_ht_ratio'] = df['CH4_w']/df['CH4_ht']
        df['psamp_pflow_ratio'] = df['psamp']/df['pflow']
        df['w_duration_ratio'] = df['CH4_w']/df['duration']

        fill_feature_ratio = ['CH4_area_ht_ratio', 'CH4_w_ht_ratio', 'psamp_pflow_ratio','w_duration_ratio']
        for i in fill_feature_ratio :
            df[i] = df[i].fillna(0)
        return df

    #=============== diff, lag, rolling
    def _diff_gen(self, df, feature, period):
        df[f'{feature}_diff_{period}'] = (df[f'{feature}'].diff(period)).fillna(0)

        return df
    #===============
    def _lag_gen (self, df, feature, period):
        df[f'{feature}_lag_{period}'] = (df[f'{feature}'].shift(period)).fillna(0)
        return df

    #===============
    def _rolling_std_gen (self, df, feature, period):
        for p in period:
            df[f'{feature}_roll_std_{p}'] = (df[f'{feature}'].rolling(window=pd.to_timedelta(p), closed='left').std()).fillna(0)# self not included, NAN->0
        return df

    #===============
    def _rolling_mean_residual_gen(self, df, feature, period):
        for p in period:
            df[f'{feature}_roll_mean_{p}'] = df[f'{feature}'].rolling(window=pd.to_timedelta(p), closed='left').mean().fillna(0) # self not included
            df[f'{feature}_residual_{p}'] = df[f'{feature}']- df[f'{feature}_roll_mean_{p}']# self not included
    #================
    def _feature_eng_apply(self, df, config):
        for opt, params in config.items():
            cols = params['cols']
            period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1

            for feature in cols:
                if opt == 'diff':
                    df = self._diff_gen(df, feature, period)
                elif opt == 'lag':
                    df = self._lag_gen(df, feature, period)
                elif opt == 'roll_std':
                    df = self._rolling_std_gen( df, feature, period)
                elif opt == 'roll_mean_res':
                    df = self._rolling_mean_residual_gen(df, feature, period)
        return df 

    def transform(self, X):
        df = X.copy(deep=False)
        df = df.set_index('datetime')
        df = self._fill_Nan(df, self.feature_cols)
        df = self._ratio_featutre_gen(df)
        df.index = pd.to_datetime(df.index)
        df = self._feature_eng_apply(df, self.feature_config)
        df = self._fill_Nan(df, self.feature_cols)

        floas_cols = df.select_dtypes(include=['float64']).columns
        df[floas_cols] = df[floas_cols].astype('float32')
        
        return df