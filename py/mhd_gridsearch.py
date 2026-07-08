import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import shap
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

param_grid = {
    'clf__n_estimators': [200, 300],
    'clf__max_depth': [7, 8],
    'clf__class_weight':['balanced'],
    'clf__min_samples_leaf':[2],
    'clf__max_features':['sqrt']
}
#================ start model training
with mlflow.start_run(run_name="RF_GridSearch") as parent_run:
    
    grid = GridSearchCV(estimator=pipe_rnd, param_grid = param_grid, cv=tscv,  scoring='average_precision', n_jobs=4, return_train_score=True)
    grid.fit(X_train_final, y_train_final)

    # print best model score
    cv_results = pd.DataFrame(grid.cv_results_)
    best_index = grid.best_index_
    mean_train = cv_results.loc[best_index, 'mean_train_score']
    mean_test = cv_results.loc[best_index, 'mean_test_score']

    print(f"Best hyparameter set 5-fold mean training AP: {mean_train:.5f}")
    print(f"Best hyparameter set 5- fold mean validation set: {mean_test:.5f}")

        
    X_test_final = tsfe.fit_transform(X_test)
    y_test_final = y_test.copy()

    #=========== plotting
    #======== generate ROC plot
    RocCurveDisplay.from_estimator(grid, X_test_final, y_test_final)
    
    plt.title("ROC Curve via RocCurveDisplay")
    plt.plot([0, 1], [0,1], 'k--') 
    plt.savefig("image/ROC_Curve.png")
    mlflow.log_artifact("image/ROC_Curve.png")
    #plt.show()    
    plt.close()
    # ======= generate PRC plot 
    PrecisionRecallDisplay.from_estimator(grid, X_test_final, y_test_final)

    plt.title("Precision-Recall Curve")
    plt.savefig("image/PRC_Curve.png")
    mlflow.log_artifact("image/PRC_Curve.png")
    #plt.show()    
    plt.close()
    #========= generate SHAP plot
    best_pipeline = grid.best_estimator_
    final_rf_model = best_pipeline.named_steps['clf']


    explainer = shap.TreeExplainer(final_rf_model)
    shap_values = explainer(X_test_final)

    shap.plots.beeswarm(shap_values[:, :, 1], max_display=46)
    plt.savefig("image/shap_summary.png")
    mlflow.log_artifact("image/shap_summary.png")
    #plt.show()    
    plt.close()

print("All Done!")