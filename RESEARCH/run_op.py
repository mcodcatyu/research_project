# data tools and plotting
from feature_eng import TSFE # feature engineering py file
import logging
import warnings
from pandas.errors import PerformanceWarning
import joblib
import optuna
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)
logging.getLogger("mlgflow.sklearn").setLevel(logging.ERROR)
from joblib.externals.loky import get_reusable_executor
import gc
#====================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
# Model from scikit-learn

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier
from sklearn.base import clone
# Model Evaluations
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.model_selection import  TimeSeriesSplit,  GridSearchCV, train_test_split
from sklearn.metrics import PrecisionRecallDisplay


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

#=====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
split = 5
N_TRIALS = 10
SEARCH_N_BAGS = 5
FINAL_N_BAGS = 30

X_tr_np = X_train_final.to_numpy()
y_tr_np = y_train.to_numpy()
X_te_np = X_test_final.to_numpy()
y_te_np = y_test.to_numpy()

def train_one_bag(base_model, X, y, pos_idx, unl_idx, number_pos, seed):
    rng = np.random.default_rng(seed)
    sampled_unl_idx = rng.choice(unl_idx, size=number_pos, replace=False)
    sampled_idx = np.concatenate([
                pos_idx,
                sampled_unl_idx
            ])
    rng.shuffle(sampled_idx)
    
    X_pu = X[sampled_idx]
    y_pu = y[sampled_idx]
    m= clone(base_model)
    m.fit(X_pu, y_pu)
    return m



def bagging_rf(n_bags, base_model, X, y, n_jobs=2, random_state=42):
    pos_idx = np.flatnonzero(y == 1)
    unl_idx = np.flatnonzero(y == 0)


    # obtain the number of positive and unlabeled data
    print(f"Number of Bag: {n_bags}")
    number_pos = len(pos_idx)
    print("Positive:", len(pos_idx))
    print("Unlabeled:", len(unl_idx))
    models = Parallel(n_jobs=n_jobs)(
        delayed(train_one_bag)(
            base_model, X, y, pos_idx, unl_idx, number_pos, seed=random_state+i
        )for i in range (n_bags)
    )
    return models

def predict_proba_pu (models, X):
    probs = [m.predict_proba(X)[:,1] for m in models]
    return np.mean(probs, axis=0)


def get_best_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        score = f1_score(
            y_true, (y_probs >= thresh).astype(int), zero_division=0
        )
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh, best_f1

def create_base_model(model_name, params):
    if model_name == "Random Forest":
        return RandomForestClassifier(**params, 
            random_state=42,
            n_jobs=1)


    elif model_name=="XGBoost":
        return XGBClassifier(**params,
            random_state=42,
            n_jobs=1
        )

    elif model_name == 'LightGBM':
        return LGBMClassifier(**params, 
            random_state=42, verbosity=-1,
            n_jobs=1    
    )

def standard_model(trial, model_name, X_tr_all, y_tr_all):
    if model_name == 'Random Forest':
        params={    
                "n_estimators":trial.suggest_int('rf_n_estimators', 50, 200, step=50),
                "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
                }
    elif model_name== 'XGBoost':
        params = {
            'n_estimators':trial.suggest_int('xgb_n_estimators', 50, 200, step=50),
            'max_depth':trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate':trial.suggest_float(
                'xgb_lr', 0.01, 0.2, log=True
            ),
        }

    elif model_name == 'LightGBM':
        params = {
            'n_estimators':trial.suggest_int('lgb_n_estimators', 50, 200, step=50),
            'max_depth':trial.suggest_int('lgb_max_depth', 3, 10),
            'learning_rate':trial.suggest_float(
                'lgb_lr', 0.01, 0.2, log=True
            ),
        }

    #==========================
    tscv = TimeSeriesSplit(n_splits = split)
    fold_f1s, fold_thresh = [], []

    for tr_idx, val_idx in tscv.split(X_tr_all):
        X_tr, X_val = X_tr_all[tr_idx], X_tr_all[val_idx]
        y_tr, y_val = y_tr_all[tr_idx], y_tr_all[val_idx]

        clf = create_base_model(model_name, params)
        clf.fit(X_tr, y_tr)

        y_prob_val = clf.predict_proba(X_val)[:,1]
        b_thresh, b_f1 = get_best_threshold(y_val, y_prob_val)

        fold_f1s.append(b_f1)
        fold_thresh.append(b_thresh)
    trial.set_user_attr('best_thresh', float(np.mean(fold_thresh)))
    return float(np.mean(fold_f1s))

def pu_model (trial, model_name, X_tr_all, y_tr_all):
    if model_name == 'Random Forest':
        params={    
                "n_estimators":trial.suggest_int('rf_n_estimators', 50, 200, step=50),
                "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
                }
    elif model_name== 'XGBoost':
        params = {
            'n_estimators':trial.suggest_int('xgb_n_estimators', 50, 200, step=50),
            'max_depth':trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate':trial.suggest_float(
                'xgb_lr', 0.01, 0.2, log=True
            ),
        }

    elif model_name == 'LightGBM':
        params = {
            'n_estimators':trial.suggest_int('lgb_n_estimators', 50, 200, step=50),
            'max_depth':trial.suggest_int('lgb_max_depth', 3, 10),
            'learning_rate':trial.suggest_float(
                'lgb_lr', 0.01, 0.2, log=True
            )
        }

    tscv = TimeSeriesSplit(n_splits=split)
    fold_f1s, fold_thresh = [], []

    for tr_idx, val_idx in tscv.split(X_tr_all):
        X_tr, X_val = X_tr_all[tr_idx], X_tr_all[val_idx]
        y_tr, y_val = y_tr_all[tr_idx], y_tr_all[val_idx]

        clf = create_base_model(model_name, params)
        pu_models = bagging_rf(SEARCH_N_BAGS, clf, X_tr, y_tr, n_jobs=2)


        y_prob_val =predict_proba_pu(pu_models, X_val)

        b_thresh, b_f1 = get_best_threshold(y_val, y_prob_val)

        fold_f1s.append(b_f1)
        fold_thresh.append(b_thresh)
    trial.set_user_attr('best_thresh', float(np.mean(fold_thresh)))

    gc.collect()
    get_reusable_executor().shutdown(wait=False)
    return float(np.mean(fold_f1s))


model_names = ["Random Forest", "XGBoost","LightGBM"]
results =[]
#fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 9))
fig_roc, ax_roc = plt.subplots(figsize=(8,6))
fig_prc, ax_prc = plt.subplots(figsize=(8,6)) # calculate PR-AUC / AP


for name in model_names:
    print("==== Start====")
    study_std = optuna.create_study(direction='maximize')
    study_std.optimize(
        lambda t : standard_model(t, name, X_tr_np, y_tr_np),
        n_trials = N_TRIALS,
    )
    best_thresh_std = study_std.best_trial.user_attrs['best_thresh']
    best_params_std = study_std.best_params

    final_std_clf = create_base_model(name, best_params_std)
    final_std_clf.fit(X_tr_np, y_tr_np)

    y_prob_std= final_std_clf.predict_proba(X_te_np)[:, 1]
    
    y_pred_std = (y_prob_std >= best_thresh_std).astype(int)

    tn_std, fp_std, fn_std, tp_std = confusion_matrix(y_test, y_pred_std).ravel()
    
    auc_std = roc_auc_score(y_test, y_prob_std)
    pr_auc_std = average_precision_score(y_test, y_prob_std)


    
    results.append({
        "Model":name, "Type":"Standard", "Threshold":best_thresh_std,
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
    prec_std, rec_std, _ = precision_recall_curve(y_test, y_prob_std)
    ax_prc.plot(rec_std, prec_std, label=f'{name} (Standard) - PR-AUC: {pr_auc_std:.3f}')

    fpr_std, tpr_std, _ = roc_curve(y_test, y_prob_std)
    ax_roc.plot(fpr_std, tpr_std, label=f"{name} (Standard) - AUC: {auc_std:.3f}")

    #cm = confusion_matrix(y_test, y_pred_std)
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[0, idx], cbar=False, xticklabels=['Pred:0','Pred:1'], yticklabels=['True:0', 'True:1'])

    #============= Pu model
    study_pu = optuna.create_study(direction='maximize')
    study_pu.optimize(
        lambda t : pu_model(t, name, X_tr_np, y_tr_np),
        n_trials=N_TRIALS
    )

    best_thresh_pu = study_pu.best_trial.user_attrs['best_thresh']
    best_params_pu = study_pu.best_params.copy()

  

    final_pu_base = create_base_model(name, best_params_pu)
    final_pu_models = bagging_rf(
        FINAL_N_BAGS, final_pu_base, X_tr_np, y_tr_np, n_jobs=-1
    )

    

    y_prob_pu = predict_proba_pu(final_pu_models, X_te_np)
    y_pred_pu = (y_prob_pu >= best_thresh_pu).astype(int)

    tn_pu, fp_pu, fn_pu, tp_pu = confusion_matrix(y_test, y_pred_pu ).ravel()
    auc_pu = roc_auc_score(y_test, y_prob_pu)
    pr_auc_pu = average_precision_score(y_test, y_prob_pu)
        
    results.append({
        "Model": name, "Type": "PU Learning", "Threshold": best_thresh_pu,
        "ROC-AUC": round(auc_pu, 4),
        "PR-AUC":round (pr_auc_pu, 4),
        "Precision": round(precision_score(y_test, y_pred_pu, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred_pu, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred_pu, zero_division=0 ), 4),

        "TN":tn_pu,
        "FP":fp_pu,
        "FN":fn_pu,
        "TP":tp_pu,
    })

    fpr_pu, tpr_pu, _ = roc_curve(y_test, y_prob_pu)
    ax_roc.plot(fpr_pu, tpr_pu, linestyle='--', label=f'{name} (PU) - AUC: {auc_pu: .3f}')

    prec_pu, rec_pu, _ = precision_recall_curve(y_test, y_prob_pu)
    ax_prc.plot(rec_pu, prec_pu, linestyle='--', label=f'{name} (PU) - PU-AUC:{pr_auc_pu: .3f}')

    #cm = confusion_matrix(y_test, y_pred_pu)
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes_cm[1, idx], cbar=False, xticklabels=['Pred:0','Pred:1'], yticklabels=['True:0', 'True:1'])
    
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


#fig_cm.tight_layout()
#fig_cm.savefig("img/PU_Confusion_Matrix_Comparison.png")


#plt.show()


df_metrics = pd.DataFrame(results)
print("======= Final results ===========")
print(df_metrics.to_string(index=False))
df_metrics.to_csv(f"img/PU_vs_Standard_Metrics_{timestamp}.csv", index=False)
#===========