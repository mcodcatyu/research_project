import numpy as np
import pandas as pd
import lightgbm as lgb
from feature_eng import TSFE
from sklearn.base import clone
import joblib

def bagging_rf(n_bags, base_model, X_train_final, y_train):
            pos_idx = np.flatnonzero(y_train.to_numpy() == 1)
            unl_idx = np.flatnonzero(y_train.to_numpy() == 0)
            # obtain the number of positive and unlabeled data
            number_pos = len(pos_idx)

            rng = np.random.default_rng(42)
            models = []
            for bag in range(n_bags):
                sampled_pos = pos_idx

                #select same number data of positive data from unlabeld data
                sampled_unl_idx = rng.choice(
                    unl_idx,
                    size=number_pos,
                    replace=False
                )

                sampled_idx = np.concatenate([
                    pos_idx,
                    sampled_unl_idx
                ])


                rng.shuffle(sampled_idx)

                X_pu = X_train_final.iloc[sampled_idx]
                y_pu = y_train.iloc[sampled_idx]
                model_clone = clone(base_model)
                model_clone.fit(X_pu, y_pu)
                models.append(model_clone)
            return models

#===========================================
df = pd.read_csv('../data/processed/mhd_ch4_cnan_v1.csv', index_col= 'datetime')
df = df.drop(df[df['year']==2026].index)

target ='label1'

feature_cols= [
  'type',
  'pflow', 'tmod',  'CH4_rt', 'CH4_w',
    'CH4_ht', 'CH4_area', 'CH4_skew', 'CH4_start_time', 'CH4_end_time',
    'CH4_start_level', 'CH4_end_level', #'duration', 
    'is_air','is_std',
    #'previous_type_std', 'previous_type_air','next_type_std', 'next_type_air', 
    ]
#
# only select the feature include in feature_cols
X = df[feature_cols]
y = df[target]

# 80% training, 20% testing(time-series data, split by time)
split_idx = int(len(df)*0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


print('Feature eng... it may take about 10 mins')

tsfe = TSFE(feature_cols=feature_cols)
X_train_final = tsfe.transform(X_train)

X_test_final = tsfe.transform(X_test)

# features we will use in model training
feature_ml =  ['CH4_ht_roll_std_24h_CH4_ht_roll_mean_24h_ratio',
 'CH4_rt_to_last_air_ratio',
 'CH4_area_roll_std_24h_CH4_area_roll_mean_24h_ratio',
 'duration_rt_ratio',
 'rt_position',
 'CH4_ht_to_last_std_ratio',
 'CH4_area_roll_std_3h_CH4_area_roll_mean_3h_ratio',
 'CH4_w_roll_std_24h_CH4_w_roll_mean_24h_ratio',
 'CH4_ht_roll_std_3h_CH4_ht_roll_mean_3h_ratio',
 'CH4_end_time',
 'CH4_w_residual_6h',
 'CH4_rt_pflow_ratio',
 'level_area_ratio',
 'CH4_w_robust_residual_3h',
 'CH4_w_residual_3h',
 'CH4_w_robust_residual_6h',
 'CH4_area_robust_residual_1h',
 'CH4_area_residual_1h',
 'CH4_area_to_last_std_ratio',
 'CH4_w_roll_std_3h_CH4_w_roll_mean_3h_ratio',
 'CH4_start_level_to_last_std_ratio',
 'CH4_area_diff_1',
 'CH4_ht_pflow_ratio',
 'CH4_ht_diff_1_CH4_ht_lag_1_per_change',
 'CH4_w_residual_24h',
 'CH4_ht_residual_1h',
 'CH4_w_diff_1',
 'CH4_w',
 'CH4_start_time',
 'CH4_area_to_last_air_ratio']

X_train_final  = X_train_final [feature_ml]
X_test_final  = X_test_final [feature_ml]
#=================

# timestamp created for figure save
model_filename = 'default_model.joblib'
n_bags=30
base_rf = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=50, random_state = 42, n_jobs=1)
pu_models= bagging_rf(n_bags, base_rf , X_train_final, y_train)

joblib.dump(pu_models, model_filename)
