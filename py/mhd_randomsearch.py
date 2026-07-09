import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import shap
from scipy.stats import randint

# Model from scikit-learn

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import PrecisionRecallDisplay

from mhd_feature import TSFE # import TSF from 'mhd_feature.py'
#=============================needed features and feature_config setting ===========================

feature_cols= [#'datetime', 
  #'tamb', , 'pamb' ,'tmod', 
  'psamp','pflow',  'CH4_rt', 'CH4_w',
       'CH4_ht', 'CH4_area', 'CH4_skew', 'CH4_start_time', 'CH4_end_time',
       'CH4_start_level', 'CH4_end_level', 'duration', 
       'is_ht_zero_and_C_Nan','is_normal_std', #'is_bad_std', 'is_protential_flagged_air'
       ]

feature_config ={
    'diff':{'cols': ['CH4_area', 'CH4_ht', #'CH4_rt'
                    ], 'periods':1},
    'lag':{'cols':['CH4_area', 'CH4_rt', #'tmod',  'psamp','pamb','tamb',
                   'pflow','CH4_w','CH4_skew'], 'periods':1},

    'roll_std':{'cols': ['CH4_w', 'CH4_skew'], 'period':'14D'},
    'roll_mean_res':{'cols':['CH4_rt', 'CH4_start_time', 'CH4_w', 'CH4_skew'], 'period': '14D'}
}
#=================== preprocessing and model fitting ============

df = pd.read_csv('../data/processed/mhd_ch4_simple_v2.csv')
df = df.drop(df[df['year']==2026].index)# the latest years'data wouldn't be used
df = df.set_index('datetime')
df = df.reset_index()

print("start running...")
# recorded by mlflow
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)
mlflow.sklearn.autolog()

mlflow.set_experiment('GridSearch_mhd_ch4')

# first 80% of the data as training set, 20% as testimg set
X = df[['datetime']+feature_cols]
y = df['label1']
split_idx = int(len(df)*0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# 5 fold time series cross-validation for hyperparameter tuning
tscv = TimeSeriesSplit(n_splits=5)

# feature engineerung
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

X_train_final = tsfe.fit_transform(X_train)
y_train_final = y_train.copy()

pipe_rnd = Pipeline([
    ('clf', RandomForestClassifier())
])

param_dist = {
    'clf__n_estimators':randint(50, 200),
    'clf__max_depth': randint(3, 12),
    'clf__class_weight':['balanced'],
    'clf__min_samples_leaf':randint(1, 10),
    'clf__max_features':['sqrt']
}
#============== start model training
with mlflow.start_run(run_name="RF_randomSearch") as parent_run:
    print('start')
    random_search= RandomizedSearchCV(
        estimator=pipe_rnd,
        param_distributions=param_dist,
        n_iter=3,
        cv = tscv,
        scoring='average_precision',
        random_state=42
    )

    random_search.fit(X_train_final, y_train_final)
print("All Done!")