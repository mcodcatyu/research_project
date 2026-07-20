# data tools and plotting

import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("mlgflow.sklearn").setLevel(logging.ERROR)
#====================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Model from scikit-learn

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import TunedThresholdClassifierCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import PrecisionRecallDisplay

import mlflow
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000);
#==========================
class TSFE(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols,  feature_config):#, target_col='label1'):
        self.feature_cols = feature_cols
        #self.target_col = target_col
        self.feature_config = feature_config
    #==========
    def fit (self, X, y=None):
        return self
    #=============
    def _fill_Nan(self, df, feature_cols):
        
        df[feature_cols] = df[feature_cols].ffill().fillna(df[feature_cols].median()) # df[feature_cols].median()
        return df
    # ============
    def _ratio_featutre_gen(self, df, feature):
        
        df[f'{feature[0]}_{feature[1]}_ratio'] = df[feature[0]]/df[feature[1]]
        return df
    #===================
    def _diff_cross_gen(self, df, feature):
    
        df[f'{feature[0]}_{feature[1]}_diff_cross'] = df[feature[0]]-df[feature[1]]
        return df
    #=============
    def _multi_featutre_gen(self, df, feature):
        df[f'{feature[0]}_{feature[1]}_multi'] = df[feature[0]]*df[feature[1]]
        return df

    #=============== diff, lag, rolling
    def _diff_gen(self, df, feature, period):
        for p in period:
            df[f'{feature}_diff_{p}'] = (df[f'{feature}'].diff(p))#.fillna(df[f'{feature}_diff_{p}'].median())

        return df
    #===============
    def _lag_gen (self, df, feature, period):
        for p in period:
            df[f'{feature}_lag_{p}'] = (df[f'{feature}'].shift(p))#.fillna(df[f'{feature}_lag_{p}'].median())
        return df
    #===============
    def _per_change_feature_gen(self, df, feature):

        df[f'{feature[0]}_{feature[1]}_per_change'] = (df[feature[0]]/df[feature[1]]+1e-7)*100
        return df

    #===============
    def _rolling_std_gen (self, df, feature, period):
        for p in period:
            df[f'{feature}_roll_std_{p}'] = (df[f'{feature}'].rolling(window=p, closed='left').std())#.fillna(df[f'{feature}_roll_std_{p}'].median())# self not included, NAN->0
        return df

    #===============
    def _roll_mean_percent_res_gen(self, df, feature, period):
        for p in period:
            df[f'{feature}_roll_mean_{p}'] = df[f'{feature}'].rolling(window=p, closed='left').mean()#.fillna(df[f'{feature}_roll_mean_{p}'].median()) # self not included
            df[f'{feature}_residual_{p}'] = ((df[f'{feature}']- df[f'{feature}_roll_mean_{p}'])/(df[f'{feature}_roll_mean_{p}']+1e-7))*100 # self not included
        return df
    #================
    def _Zscore_res_gen(self, df, feature):
        df[f'{feature[0]}_{feature[3]}_zcore_res_gen'] = ((df[feature[0]]-df[feature[1]])/df[feature[2]]*1e-7)
        return df
    #=================
    def _log_gen (self, df, feature):
        df[f'{feature}_log'] = np.sign(df[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))#.fillna(df[f'{feature}_lag_{p}'].median())
        return df
    #===========
    def _relative_per_gen(self, df, feature, period):
        for p in period:
            df[f'{feature}_relative_per_{p}'] = (df[f'{feature}']-(df[f'{feature}'].rolling(window=p, closed='left').min()))/((df[f'{feature}'].rolling(window=p, closed='left').max())-(df[f'{feature}'].rolling(window=p, closed='left').min())+1e-7) #.fillna(df[f'{feature}_roll_std_{p}'].median())# self not included, NAN->0
        return df
    #=================
    def _per_rank_gen(self, df, feature, period):
        for p in period:
            df[f'{feature}_per_rank_{p}'] = df[f'{feature}'].rolling(window=p, closed='left').rank(pct=True)
        return df
    #=================
    def _feature_eng_apply(self, df, config):
        for opt, params in config.items():
            cols = params['cols']
           # period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1

            for feature in cols:
                if opt == 'diff':
                    period = params.get('period') or params.get('periods', 1)
                    df = self._diff_gen(df, feature, period)
                elif opt == 'ratio':
                    df = self._ratio_featutre_gen(df, feature)
                elif opt == 'diff_cross':
                    df = self._diff_cross_gen(df, feature)
                elif opt == 'multi':
                    df = self._multi_featutre_gen(df, feature)
                elif opt == 'lag':
                    period = params.get('period') or params.get('periods', 1)
                    df = self._lag_gen(df, feature, period)
                elif opt == 'per_change':
                    df = self._per_change_feature_gen(df, feature)
                elif opt == 'roll_std':
                    period = params.get('period') or params.get('periods', 1)
                    df = self._rolling_std_gen( df, feature, period)
                elif opt == 'roll_mean_percent_res':
                    period = params.get('period') or params.get('periods', 1)
                    df = self._roll_mean_percent_res_gen(df, feature, period)
                elif opt == 'Z_score_res':
                    df = self._Zscore_res_gen(df, feature)
                elif opt == 'log':
                    df = self._log_gen(df, feature)
                elif opt  =='relative_per':
                    period = params.get('period') or params.get('periods', 1)
                    df = self._relative_per_gen(df, feature, period)
                elif opt == 'per_rank':
                    period = params.get('period') or params.get('periods', 1)
                    df = self._per_rank_gen(df, feature, period)

        return df 

    def transform(self, X):
        df = X.copy(deep=False)
        #df = self._fill_Nan(df, self.feature_cols)
        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values
        #df = self._ratio_featutre_gen(df, self.feature_cols)
        df.index = pd.to_datetime(df.index)
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values
        df_columns = df.columns.tolist()
        df = self._fill_Nan(df, df_columns)

        floas_cols = df.select_dtypes(include=['float64']).columns
        df[floas_cols] = df[floas_cols].astype('float32')

        return df
    

#=====================

from sklearn.metrics import precision_recall_curve
def get_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    p = precisions[:-1]
    r = recalls[:-1]

    f1_scores = 2 * (p * r) / (p + r + 1e-7) # 1e-7 to prevent Nan

    best_index = np.argmax(f1_scores)
    best_thres = thresholds[best_index]
    return best_thres, f1_scores[best_index], p[best_index], r[best_index]

#=========mlflow setting==========


mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)


mlflow.set_experiment('isCnan_mhd')
#=============== feature engineering and define the needed features

feature_cols= [#'datetime', 
  #'tamb',  'pamb' ,
  #'psamp',
  
  'pflow', 'tmod',  'CH4_rt', 'CH4_w',
       'CH4_ht', 'CH4_area', 'CH4_skew', 'CH4_start_time', 'CH4_end_time',
       'CH4_start_level', 'CH4_end_level', #'duration', 
       'is_air','is_std',
       #'previous_type_std', 'previous_type_air','next_type_std', 'next_type_air', 
       'last_std_ht', 'last_air_ht','level_rt_ratio',
       'rt_position',
        'baseline_slope',
        'level_area_ratio',
        'last_std_area',
        'last_air_area',
        'last_std_rt',
        'last_air_rt',
        'last_std_start_level',
        'last_air_start_level',
       #'is_ht_zero_and_C_Nan','is_normal_std', 
       # #'is_bad_std', 'is_protential_flagged_air'
       ]



feature_config ={
    'diff':{'cols': ['CH4_area', 
                    'CH4_ht','CH4_w' ,#'tmod',#'CH4_end_time','CH4_start_time',,'CH4_rt'
                ], 'periods':[1]},

    'lag':{'cols':['CH4_area', 'CH4_rt',  'CH4_ht', #'tmod',  'psamp','pamb','tamb',,'CH4_skew'
                #'pflow',
                'CH4_w'], 'periods':[1]},

    

    'roll_std':{'cols': ['CH4_w', 'CH4_end_time','CH4_ht','CH4_area'], 'period':['2h','6h','24h','2D']},

    'roll_mean_percent_res':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area'], 'period': ['2h','6h','24h','2D']},

    'diff_cross':{'cols':[['CH4_end_time', 'CH4_start_time'],
                          # ['CH4_w_roll_mean_2h', 'CH4_w_roll_mean_7h'],
                           ['last_std_ht', 'last_air_ht'],#['CH4_end_level', 'CH4_start_level'],
                           ['CH4_ht', 'last_std_ht'], ['CH4_ht', 'last_air_ht'],#['CH4_ht', 'CH4_ht_lag_72'],['CH4_w', 'CH4_w_lag_72'],
                           #['pflow', 'pflow_lag_1'],['CH4_skew','CH4_skew_lag_1']
                           ]}, 

    'log':{'cols':['last_std_ht','last_air_ht'
                   #'CH4_end_time', 
                   ]},

    'multi':{'cols':[['CH4_w', 'CH4_ht'],['CH4_end_time_CH4_start_time_diff_cross','last_std_ht_log'], ['CH4_end_time_CH4_start_time_diff_cross','last_air_ht_log']]},

    'ratio':{'cols':[   
                        ['CH4_area','CH4_w_CH4_ht_multi'], ['CH4_ht', 'CH4_area'],
                        ['CH4_w', 'CH4_end_time_CH4_start_time_diff_cross'],
                       # ['CH4_rt', 'CH4_end_time_CH4_start_time_diff_cross'],

                        ['CH4_ht', 'last_air_ht'],# ['CH4_ht', 'last_air_ht'],
                        #['CH4_w_roll_std_2h', 'CH4_w_roll_std_1D'],
                        ['last_std_ht', 'last_air_ht'],
                        ['CH4_start_level', 'CH4_end_level'],
                        ['CH4_area', 'last_std_area'],['CH4_start_level', 'last_std_start_level'],
                        
                        ['CH4_w_roll_std_2h', 'CH4_w_roll_mean_2h'], ['CH4_ht_roll_std_2h', 'CH4_ht_roll_mean_2h'],['CH4_area_roll_std_2h', 'CH4_area_roll_mean_2h'],
                        ['CH4_w_roll_std_6h', 'CH4_w_roll_mean_6h'], ['CH4_ht_roll_std_6h', 'CH4_ht_roll_mean_6h'],['CH4_area_roll_std_6h', 'CH4_area_roll_mean_6h'],
                        #['CH4_ht_roll_std_24h', 'CH4_ht_roll_mean_24h']
                       # ['CH4_w_roll_std_7h', 'CH4_w_roll_mean_7h'],['CH4_ht_roll_std_7h', 'CH4_ht_roll_mean_7h'],
                        #['CH4_w_roll_std_2D', 'CH4_w_roll_mean_2D'],['CH4_ht_roll_std_2D', 'CH4_ht_roll_mean_2D'],
                        #['CH4_rt', 'last_std_rt'],
                        
                       #['CH4_end_time', 'CH4_start_time'], ['CH4_area', 'pflow'],  ['CH4_skew', 'CH4_w'],['CH4_w', 'CH4_ht']
                       ]}, 


    #'per_change':{'cols':[['CH4_ht_diff_1', 'CH4_ht_lag_1'],['last_std_ht', 'last_air_ht']]},
    #'relative_per': {'cols': ['CH4_w', 'CH4_end_time','CH4_ht'], 'period':['2h','7h']},
    #'Z_score_res':{'cols':[['CH4_w', 'CH4_w_roll_mean_2h', 'CH4_w_roll_std_2h', '2h'], #['CH4_end_time', 'CH4_end_time_roll_mean_7D', 'CH4_end_time_roll_std_7D', '7D']
                           #]},
    #'per_rank': {'cols': ['CH4_w', 'CH4_skew','CH4_ht'], 'period':['7h','7D','30D']},


}
#========= mlfeature
feature_ml = ['CH4_w_roll_std_2h_CH4_w_roll_mean_2h_ratio', 
              'CH4_w_roll_std_6h_CH4_w_roll_mean_6h_ratio', 
              'CH4_area_roll_std_6h_CH4_area_roll_mean_6h_ratio', 
              'CH4_end_time_CH4_start_time_diff_cross_last_std_ht_log_multi', 
              'CH4_area_roll_std_2h_CH4_area_roll_mean_2h_ratio', 
              'CH4_ht_roll_std_2h_CH4_ht_roll_mean_2h_ratio', 
              'CH4_w_residual_6h', 
              'CH4_w_diff_1',
                'CH4_ht_roll_std_6h_CH4_ht_roll_mean_6h_ratio', 
               'CH4_w_residual_2h', 
               'level_area_ratio',
                'CH4_area_last_std_area_ratio', 
                'CH4_end_time_CH4_start_time_diff_cross',
                'CH4_ht_roll_mean_2h', 
                'CH4_end_time', 
                'CH4_w_roll_std_2h', 
                'CH4_w_diff_2',
                'CH4_w_roll_mean_2h', 
                'CH4_rt_roll_mean_2h', 
                'CH4_end_time_CH4_start_time_diff_cross_last_air_ht_log_multi',
                'CH4_w_roll_std_6h', 
                'CH4_start_level_last_std_start_level_ratio', 
                'CH4_w_CH4_end_time_CH4_start_time_diff_cross_ratio', 
                'rt_position',
                'level_rt_ratio']

#================ data reading and preprocessing=========
df = pd.read_csv('../data/processed/mhd_ch4_cnan_v1.csv', index_col= 'datetime')
df = df.drop(df[df['year']==2026].index)

df.columns.tolist()
#=====================ht========================
df['only_std_ht'] = df['CH4_ht'].where(df['type'] == 'std')
std_median = df.loc[df['type']=='std', 'CH4_ht'].median()

df['last_std_ht'] = df['only_std_ht'].ffill().shift(1).fillna(std_median)
#======== air
df['only_air_ht'] = df['CH4_ht'].where(df['type'] == 'air')
air_median = df.loc[df['type']=='air', 'CH4_ht'].median()
df['last_air_ht'] = df['only_air_ht'].ffill().shift(1).fillna(air_median)
#df['is_CH4_w_nan'] = df['CH4_w'].isna().astype(int) 
#=====================
df['next_std_ht'] = df['only_std_ht'].bfill().shift(-1).fillna(std_median)


df['next_air_ht'] = df['only_air_ht'].bfill().shift(-1).fillna(air_median)
#=================== area========================
df['only_std_area'] = df['CH4_area'].where(df['type'] == 'std')
std_median = df.loc[df['type']=='std', 'CH4_area'].median()

df['last_std_area'] = df['only_std_area'].ffill().shift(1).fillna(std_median)
#======== air
df['only_air_area'] = df['CH4_area'].where(df['type'] == 'air')
air_median = df.loc[df['type']=='air', 'CH4_area'].median()
df['last_air_area'] = df['only_air_area'].ffill().shift(1).fillna(air_median)

#====================== rt =============================
df['only_std_rt'] = df['CH4_rt'].where(df['type'] == 'std')
std_median = df.loc[df['type']=='std', 'CH4_rt'].median()

df['last_std_rt'] = df['only_std_rt'].ffill().shift(1).fillna(std_median)

df['only_air_rt'] = df['CH4_rt'].where(df['type'] == 'air')
air_median = df.loc[df['type']=='air', 'CH4_rt'].median()
df['last_air_rt'] = df['only_air_rt'].ffill().shift(1).fillna(air_median)


#==========================start level===================
df['only_std_start_level'] = df['CH4_start_level'].where(df['type'] == 'std')
std_median = df.loc[df['type']=='std', 'CH4_start_level'].median()

df['last_std_start_level'] = df['only_std_start_level'].ffill().shift(1).fillna(std_median)
#========rt
df['only_air_start_level'] = df['CH4_start_level'].where(df['type'] == 'air')
air_median = df.loc[df['type']=='air', 'CH4_start_level'].median()
df['last_air_start_level'] = df['only_air_start_level'].ffill().shift(1).fillna(air_median)

#============= new features==============================
#df['is_CH4_w_nan'] = df['CH4_w'].isna().astype(int) 
df['level_rt_ratio'] = (df['CH4_end_time'] -df['CH4_start_time'])/df['CH4_rt']
df['rt_position'] = (df['CH4_rt'] - df['CH4_start_time']) / df['CH4_w']
df['baseline_slope'] =(df['CH4_end_level'] -df['CH4_start_level'])/df['CH4_w']
df['level_area_ratio'] = np.maximum(df['CH4_end_level'],df['CH4_start_level'] )/(df['CH4_area'])