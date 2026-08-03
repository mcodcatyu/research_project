# data tools and plotting

from feature_eng import TSFE
#====================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Model from scikit-learn

from sklearn.ensemble import RandomForestClassifier

# Model Evaluations


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import TunedThresholdClassifierCV, TimeSeriesSplit, RandomizedSearchCV

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import PrecisionRecallDisplay

import mlflow
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000);
#=============== feature engineering and define the needed features

feature_cols= [#'datetime', 
  #'tamb',  'pamb' ,
  #'psamp',
  'pflow', 'tmod',  'CH4_rt', 'CH4_w',
  'CH4_ht', 'CH4_area', 'CH4_skew', 'CH4_start_time', 'CH4_end_time',
  'CH4_start_level', 'CH4_end_level', #'duration', 
  'is_air','is_std',
  'last_std_CH4_ht','last_air_CH4_ht','last_std_CH4_area','last_air_CH4_area','last_std_CH4_rt',
  'last_air_CH4_rt','last_std_CH4_start_level',
  'last_air_CH4_start_level','CH4_ht_to_last_std_ratio',
  'CH4_ht_to_last_air_ratio','CH4_area_to_last_std_ratio',
  'CH4_area_to_last_air_ratio','CH4_rt_to_last_std_ratio',
  'CH4_rt_to_last_air_ratio', 'CH4_start_level_to_last_std_ratio', 'CH4_start_level_to_last_air_ratio',
  'duration_rt_ratio','rt_position',
  'baseline_slope','level_area_ratio'
]

feature_config ={
    'diff':{'cols': ['CH4_area', 
                    'CH4_ht','CH4_w' , 'CH4_rt' #'tmod',#'CH4_end_time','CH4_start_time',,'CH4_rt'
                ], 'periods':[1, 3, 6, 72]},

    'lag':{'cols':['CH4_area', 'CH4_rt',  'CH4_ht', #'tmod',  'psamp','pamb','tamb',,'CH4_skew'
                'pflow',
              'CH4_w'], 'periods':[1, 3, 6, 72]},

    'diff_cross':{'cols':[['CH4_ht', 'CH4_ht_lag_72'],
                          ['CH4_area', 'CH4_area_lag_72'],
                          ['CH4_end_time', 'CH4_start_time'],
                        ['CH4_end_level', 'CH4_start_level'],
                        ['CH4_rt', 'pflow'],
                        ]}, 

    'roll_std':{'cols': ['CH4_w', 'CH4_ht','CH4_area', 'CH4_end_time_CH4_start_time_diff_cross'], 'period':['1h','3h','6h', ]},

    'roll_mean_percent_res':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area',], 'period': ['1h','3h','6h',]},



    #'log':{'cols':['last_std_CH4_ht','last_air_CH4_ht'
                   #'CH4_end_time', 
     #              ]},
     'roll_median':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area', 'pflow','CH4_area_lag_72','CH4_ht_lag_72'], 'period':['1h','3h','6h',]},
     'roll_mad':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area', 'pflow'], 'period':['1h','3h','6h',]},
     'roll_median_percent_res':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area','pflow','CH4_area_lag_72','CH4_ht_lag_72'],
                              'period':['1h','3h','6h', '24h', '3D', '7D']},

   # 'multi':{'cols':[['CH4_w', 'CH4_ht'],['CH4_area', 'pflow']]},

    'ratio':{'cols':[   ['CH4_skew','CH4_w'],
                        ['CH4_ht', 'pflow'],
                        ['CH4_ht', 'CH4_w'],
                        ['CH4_area', 'pflow'],
                        ['CH4_rt', 'pflow'],
                        ['CH4_ht', 'CH4_area'],
                        ['CH4_w', 'CH4_end_time_CH4_start_time_diff_cross'],
                        ['CH4_start_level', 'CH4_end_level'],
                  
                        ['CH4_w_roll_std_3h', 'CH4_w_roll_mean_3h'], ['CH4_ht_roll_std_3h', 'CH4_ht_roll_mean_3h'],['CH4_area_roll_std_3h', 'CH4_area_roll_mean_3h'],
                        #['CH4_w_roll_std_24h', 'CH4_w_roll_mean_24h'], ['CH4_ht_roll_std_24h', 'CH4_ht_roll_mean_24h'],['CH4_area_roll_std_24h', 'CH4_area_roll_mean_24h'],
                       ]}, 

    'per_change':{'cols':[['CH4_ht_diff_1', 'CH4_ht_lag_1']]},
    'relative_per': {'cols': ['CH4_w', 'CH4_ht'], 'period':['2h','7h']},
    'Z_score_res':{'cols':[['CH4_w', 'CH4_w_roll_mean_3h', 'CH4_w_roll_std_3h', '3h'], #['CH4_end_time', 'CH4_end_time_roll_mean_7D', 'CH4_end_time_roll_std_7D', '7D']
                           ]},
    'per_rank': {'cols': ['CH4_w', 'CH4_skew','CH4_ht'], 'period':['7D']},
}

#==============
df = pd.read_csv('../data/processed/mhd_ch4_cnan_v1.csv', index_col= 'datetime')
df = df.drop(df[df['year']==2026].index)

#df.columns.tolist()

target_cols = ['ht', 'area', 'rt', 'start_level']

for col in target_cols:
    col = f'CH4_{col}'

    for t_type in ['std', 'air']:
        type_median = df.loc[df['type'] == t_type, col ].median()

        only_series = df[col].where(df['type']==t_type)
        df[f'last_{t_type}_{col}'] = only_series.ffill().shift(1).fillna(type_median)

for col in ['CH4_ht', 'CH4_area', 'CH4_rt', 'CH4_start_level']:
    df[f'{col}_to_last_std_ratio'] = df[col] / (df[f'last_std_{col}'] + 1e-7)
    df[f'{col}_to_last_air_ratio'] = df[col] / (df[f'last_air_{col}'] + 1e-7)

df['duration_rt_ratio'] = (df['CH4_end_time'] -df['CH4_start_time'])/df['CH4_rt']
df['rt_position'] = (df['CH4_rt'] - df['CH4_start_time']) / df['CH4_w']
df['baseline_slope'] =(df['CH4_end_level'] -df['CH4_start_level'])/df['CH4_w']
df['level_area_ratio'] = np.maximum(df['CH4_end_level'],df['CH4_start_level'] )/(df['CH4_area']+1e-7)
#======
target ='label1'
#======
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from pulearn import ElkanotoPuClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

X = df[feature_cols]
y = df[target]

tscv = TimeSeriesSplit(n_splits=5)

folds_data = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
      
    tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

                
    X_train_final = tsfe.fit_transform(X_train)
    #y_train_final = y_train.copy()

    #selected_features = slected_col(X_train_final)
    #X_train_final  = X_train_final [feature_ml]

    X_test_final = tsfe.transform(X_test)
    #X_test_final  = X_test_final [feature_ml]

    folds_data.append({
        'X_train': X_train_final,
        'y_train': y_train.copy(),
        'X_test':X_test_final,
        'y_test':y_test.copy()
    })

def evaluate_percentage(p, folds_data, base_seed=42):
    pu_fold_scores = []
    reg_fold_scores = []

    p_seed = int(base_seed + p*1000)

    for i, fold in enumerate(folds_data):
        X_train_final = fold['X_train']
        y_train_pu = fold['y_train'].copy()
        X_test_final = fold['X_test']
        y_test_final = fold['y_test']
        pos_in_fold = y_train_pu[y_train_pu == 1.0].index.tolist()
        n_sacrifice = int(len(pos_in_fold)*p)

        rng = np.random.default_rng(p_seed + i)
        sacrifice_indices = rng.choice(pos_in_fold, size=n_sacrifice, replace=False)
        y_train_pu.loc[sacrifice_indices] = 0.0

        estimator = RandomForestClassifier(
                            n_estimators=50,
                            criterion="gini",
                            bootstrap=True,
                            n_jobs=1,
                            random_state=p_seed + i
                    )
        pu_estimator = ElkanotoPuClassifier(estimator)
        pu_estimator.fit(X_train_final, y_train_pu)

        y_pred_pu = pu_estimator.predict(X_test_final)
        pu_f1 = f1_score(y_test_final, y_pred_pu, pos_label=1.0)
        
        pu_fold_scores.append(pu_f1)
        #====================================
        reg_model = RandomForestClassifier(n_estimators=50, random_state=42)
        reg_model.fit(X_train_final, y_train_pu)

        y_pred_reg = reg_model.predict(X_test_final)
        reg_f1 = f1_score(y_test_final, y_pred_reg, pos_label=1.0)
        reg_fold_scores.append(reg_f1)
    mean_pu_f1 = np.mean(pu_fold_scores)
    mean_reg_f1 = np.mean(reg_fold_scores)

    print(f"Finish percentage: {p*100:3.0f}%, | PU F1:{mean_pu_f1:.4f}| Reg F1: {mean_reg_f1:.4f}")
    return p, mean_pu_f1, mean_reg_f1



percentages = np.linspace(0, 0.9, 10)
print("------- Parallel Pu learning experement-----")

results = Parallel(n_jobs=4)(
    delayed(evaluate_percentage)(p, folds_data) for p in percentages
)

results.sort(key=lambda x:x[0])
pu_f1_scores= [r[1] for r in results]
reg_f1_scores = [r[2] for  r in results]

plt.figure(figsize=(8,6))
plt.title("Random forest with/without PU learning")
plt.plot(percentages*100, pu_f1_scores, marker='o', label="PU Adapted Random Forest")
plt.plot(percentages*100, reg_f1_scores, marker='s', label="Random Forest")
plt.xlabel("Percentage of positive examples hidden in the unlabeled set")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylabel("F1 Score")
plt.legend()
plt.savefig('pu_nonpu_compar.png')