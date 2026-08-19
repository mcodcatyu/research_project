# data tools and plotting
from feature_eng import TSFE # feature engineering py file
import logging
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)
logging.getLogger("mlgflow.sklearn").setLevel(logging.ERROR)
from datetime import datetime
#====================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
# Model from scikit-learn

from sklearn.ensemble import RandomForestClassifier

import  xgboost as xgb
import lightgbm as lgb
from sklearn.base import clone
# Model Evaluations
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.model_selection import  TimeSeriesSplit,  GridSearchCV, train_test_split
from sklearn.metrics import PrecisionRecallDisplay


os.makedirs('img', exist_ok=True) 

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

X = df[feature_cols]
y = df[target]
split_idx = int(len(df)*0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


tscv = TimeSeriesSplit(n_splits=5)

tsfe = TSFE(feature_cols=feature_cols)
X_train_final = tsfe.transform(X_train)

X_test_final = tsfe.transform(X_test)
feature_ml =  ['CH4_ht_roll_std_24h_CH4_ht_roll_mean_24h_ratio', 'CH4_rt_to_last_air_ratio', 
               'CH4_area_roll_std_24h_CH4_area_roll_mean_24h_ratio', 'duration_rt_ratio', 
               'rt_position', 'CH4_ht_to_last_std_ratio', 'CH4_area_roll_std_3h_CH4_area_roll_mean_3h_ratio', 
               'CH4_w_roll_std_24h_CH4_w_roll_mean_24h_ratio', 'CH4_ht_roll_std_3h_CH4_ht_roll_mean_3h_ratio', 
               'CH4_end_time', 'CH4_w_residual_6h', 'CH4_rt_pflow_ratio', 'level_area_ratio', 'CH4_w_robust_residual_6h',
                 'CH4_area_robust_residual_1h', 'CH4_area_residual_1h', 'CH4_w_roll_std_3h_CH4_w_roll_mean_3h_ratio', 
                 'CH4_area_to_last_std_ratio', 'CH4_start_level_to_last_std_ratio', 'CH4_area_diff_1']


X_train_final  = X_train_final [feature_ml]
X_test_final  = X_test_final [feature_ml]
#=================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def bagging_rf(n_bags, base_model, X_train_final, y_train):
    pos_idx = np.flatnonzero(y_train.to_numpy() == 1)
    unl_idx = np.flatnonzero(y_train.to_numpy() == 0)

    # obtain the number of positive and unlabeled data
    print(f"Number of Bag: {n_bags}")
    number_pos = len(pos_idx)
    print("Positive:", len(pos_idx))
    print("Unlabeled:", len(unl_idx))

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

        #打亂順序
        rng.shuffle(sampled_idx)

        X_pu = X_train_final.iloc[sampled_idx]
        y_pu = y_train.iloc[sampled_idx]

        print(
            f"\n PU training data, Bag {bag +1}/{n_bags} "
            f"Positive = {(y_pu ==1).sum()}, "
            f"Unlabeled = {(y_pu == 0).sum()}"
        )
        model_clone = clone(base_model)
        model_clone.fit(X_pu, y_pu)
        models.append(model_clone)
    return models


def predict_proba_pu (models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)

def get_best_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.02):
        score = f1_score(
            y_true, (y_probs >= thresh).astype(int), zero_division=0
        )
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh

#=================
param_grids = {
    "Random Forest":{
        'n_estimators':[50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf':[1,2,4]
        },
    "XGBoost":{
        'n_estimators':[50, 100, 200],
        'max_depth':[3,6, 10],
        'learning_rate':[0.01, 0.1, 0.2],
        'subsample':[0.7, 0.8, 1.0],
        },
    "LightGBM":{
        'n_estimators':[50, 100, 200],
        'num_leaves':[31,50],
        'learning_rate':[0.05, 0.1, 0.2],
    }
}
base_models = {
    "Random Forest":RandomForestClassifier(random_state=42, n_jobs=1),
    "XGBoost":xgb.XGBClassifier(random_state=42, n_jobs=1),
    "LightGBM":lgb.LGBMClassifier(random_state = 42, n_jobs=1)
} 

threshold = 0.0
results =[]
fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 9))
fig_roc, ax_roc = plt.subplots(figsize=(8,6))
fig_prc, ax_prc = plt.subplots(figsize=(8,6)) # calculate PR-AUC / AP
best_fitted_models = {}
best_params_dict = {}
best_score_dict = {}


for idx, (name, base_clf) in enumerate(base_models.items()):
    print(f" Grid Search:{name}")
    sub_ratio = 0.2
    sub_size = int(len(X_train_final)*sub_ratio)

    X_sub = X_train_final.iloc[-sub_size:]
    y_sub = y_train.iloc[-sub_size:]

    grid_search = GridSearchCV(
        estimator = clone(base_clf),
        param_grid = param_grids[name],
        cv = tscv,
        scoring = 'average_precision',
        n_jobs=-1,
        verbose=2
        
    )
    grid_search.fit(X_train_final, y_train)

    best_params= grid_search.best_params_
    best_params_dict[name] = grid_search.best_params_
    best_score_dict[name] = grid_search.best_score_
    
    print(f"Best parameter combination: {best_params}")
    best_std_clf = clone(base_clf)
    best_std_clf.set_params(**best_params)
    best_std_clf.fit(X_train_final, y_train)

    best_fitted_models[name] = best_std_clf
    y_prob_train_std= best_std_clf.predict_proba(X_train_final)[:, 1]

    threshold_std = get_best_threshold(y_train, y_prob_train_std)
    y_prob_std = best_std_clf.predict_proba(X_test_final)[:, 1]
    y_pred_std = (y_prob_std >= threshold_std).astype(int)

    auc_std = roc_auc_score(y_test, y_prob_std)
    pr_auc_std = average_precision_score(y_test, y_prob_std)

    prec_std, rec_std, _ = precision_recall_curve(y_test, y_prob_std)
    ax_prc.plot(rec_std, prec_std, label=f'{name} (Standard) - PR-AUC: {pr_auc_std:.3f}')

    tn_std, fp_std, fn_std, tp_std = confusion_matrix(y_test, y_pred_std).ravel()

    results.append({
        "Model":name, "Type":"Standard", "Threshold":round(threshold_std,4),
        "ROC-AUC": round(auc_std, 4),
        "PR-AUC": round(pr_auc_std, 4),
        "Precision": round(precision_score(y_test, y_pred_std, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred_std, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred_std, zero_division=0), 4),
        "TN":tn_std,
        "FP":fp_std,
        "FN":fn_std,
        "TP":tp_std
    })

    fpr_std, tpr_std, _ = roc_curve(y_test, y_prob_std)
    ax_roc.plot(fpr_std, tpr_std, label=f"{name} (Standard) - AUC: {auc_std:.3f}")

    cm = confusion_matrix(y_test, y_pred_std)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0, idx], cbar=False, xticklabels=['Pred:0','Pred:1'], yticklabels=['True:0', 'True:1'])

    axes_cm[0, idx].set_title(f"{name} (Standard)")
    axes_cm[0, idx].set_xlabel(f"Predicted Label")
    axes_cm[0, idx].set_ylabel(f"True Label")

    #============= Pu model
    n_bags= 30
    pu_clf= bagging_rf(n_bags,clone(best_std_clf), X_train_final, y_train)
    
    y_prob_train_pu = predict_proba_pu(pu_clf, X_train_final)
    threshold_pu = get_best_threshold(y_train, y_prob_train_pu)

    y_prob_pu = predict_proba_pu(pu_clf, X_test_final)
    y_pred_pu = (y_prob_pu >= threshold_pu).astype(int)

    auc_pu = roc_auc_score(y_test, y_prob_pu)
    pr_auc_pu = average_precision_score(y_test, y_prob_pu)
    tn_pu, fp_pu, fn_pu, tp_pu = confusion_matrix(y_test, y_pred_pu ).ravel()

    results.append({
        "Model": name, "Type": "PU Learning", "Threshold":round(threshold_pu, 4),
        "ROC-AUC": round(auc_pu, 4),
        "PR-AUC":round (pr_auc_pu, 4),
        "Precision": round(precision_score(y_test, y_pred_pu, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred_pu, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred_pu, zero_division=0 ), 4),
        "TN":tn_pu,
        "FP":fp_pu,
        "FN":fn_pu,
        "TP":tp_pu
    })

    fpr_pu, tpr_pu, _ = roc_curve(y_test, y_prob_pu)
    ax_roc.plot(fpr_pu, tpr_pu, linestyle='--', label=f'{name} (PU) - AUC: {auc_pu: .3f}')

    prec_pu, rec_pu, _ = precision_recall_curve(y_test, y_prob_pu)
    ax_prc.plot(rec_pu, prec_pu, linestyle='--', label=f'{name} (PU) - PU-AUC:{pr_auc_pu: .3f}')

    cm = confusion_matrix(y_test, y_pred_pu)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes_cm[1, idx], cbar=False, xticklabels=['Pred:0','Pred:1'], yticklabels=['True:0', 'True:1'])
    axes_cm[1, idx].set_title(f"{name} (PU)")
    axes_cm[1, idx].set_xlabel("Predicted Label")
    axes_cm[1, idx].set_ylabel("True Label")



ax_roc.plot([0, 1], [0,1], 'k--', alpha=0.5)
ax_roc.set_title("ROC CURVE: Standard vs PU learning")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()

fig_roc.tight_layout()
fig_roc.savefig(f"img/PU_ROC_Comparison_{timestamp}.png")



baseline_pr = np.sum(y_test == 1) / len(y_test)
ax_prc.axhline(y=baseline_pr, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_pr:.3f})')
ax_prc.set_title("Precision-Recall Curve: Standard vs PU Learning")
ax_prc.set_xlabel("Recall")
ax_prc.set_ylabel("Precision")
ax_prc.legend()
fig_prc.tight_layout()
fig_prc.savefig(f"img/PU_PRC_Comparison_{timestamp}.png")


fig_cm.tight_layout()
fig_cm.savefig(f"img/PU_Confusion_Matrix_Comparison_{timestamp}.png")


plt.show()


df_metrics = pd.DataFrame(results)
print(df_metrics.to_string(index=False))
df_metrics.to_csv(f"img/PU_vs_Standard_Metrics_{timestamp}.csv", index=False)
#===========
