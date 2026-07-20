import shap


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

#===================
target = 'label_c_nan'
X = df[feature_cols]
y = df[target]
split_idx = int(len(df)*0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


tscv = TimeSeriesSplit(n_splits=5)
X_train.head()


tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

X_train_final = tsfe.fit_transform(X_train)
y_train_final = y_train.copy()
print(X_train_final.columns)
X_train_final = X_train_final[feature_ml]

pipe_rnd = Pipeline([
    #('scl', StandardScaler()),
    ('clf', RandomForestClassifier())
])

param_grid = {
    'clf__n_estimators': [200],
    'clf__max_depth': [5, 8],
    'clf__min_samples_leaf':[15],
    'clf__random_state':[42],
    'clf__max_features':['sqrt'],
    'clf__n_jobs':[4],
    'clf__class_weight':['balanced']
}


with mlflow.start_run(run_name="RF_GridSearch_0720_v1") as parent_run:
    dataset_metadat={
        'feature_cols': feature_cols,
        'feature_config': feature_config,
        #'feature_ml':feature_ml
    }
    mlflow.log_dict(dataset_metadat, 'feature_config.json')
    
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
    X_test_final = X_test_final[feature_ml]
    #=========== Evaluation Report
    y_prob = grid.predict_proba(X_test_final)[:, 1]
    best_threshold, best_f1, best_prec, best_rec = get_best_threshold(y_test_final, y_prob)
    y_proba_best = (y_prob >= best_threshold).astype(int)

    print("====Evaluation report best threshold ===")
    print(f'Best threshold:{best_threshold:.4f}')
    print(f'Best F1-score:{best_f1:.4f}; Best Precision:{best_prec:.4f}, Recall:{best_rec:.4f}')
    print(classification_report(y_test_final, y_proba_best))
    #=========== plotting
    #======== ROC
    RocCurveDisplay.from_estimator(grid, X_test_final, y_test_final)
    
    plt.title("ROC Curve via RocCurveDisplay")
    plt.plot([0, 1], [0,1], 'k--') 
    plt.savefig("image/ROC_Curve.png")
    mlflow.log_artifact("image/ROC_Curve.png")
    plt.show()    
    plt.close()
    # ======= PRC 
    PrecisionRecallDisplay.from_estimator(grid, X_test_final, y_test_final)

    plt.title("Precision-Recall Curve")
    plt.savefig("image/PRC_Curve.png")
    mlflow.log_artifact("image/PRC_Curve.png")
    plt.show()    
    plt.close()
    #==========feature importance
    best_clf = grid.best_estimator_.named_steps['clf']

    feature_names = X_train_final.columns
    importances = best_clf.feature_importances_

    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_imp_df.head(20))
    print(feature_imp_df['Feature'].head(25).tolist())
    #========= SHAP
    #best_pipeline = grid.best_estimator_
    #final_rf_model = best_pipeline.named_steps['clf']


    #explainer = shap.TreeExplainer(final_rf_model)
    #shap_values = explainer(X_test_final)

    #shap.plots.beeswarm(shap_values[:, :, 1], max_display=72)
    #plt.savefig("image/shap_summary.png")
    #mlflow.log_artifact("image/shap_summary.png")
    #plt.show()    
    #plt.close()


print("All Done!")
    
best_threshold
y_prob = grid.predict_proba(X_test_final)[:, 1]
y_proba_best = (y_prob >= best_threshold).astype(int)

analysis_df = pd.DataFrame({
    'original_index': X_test_final.index,
    'true_label': y_test.values,
    'predicted_label': y_proba_best,
    'predicted_prob': y_prob
}, index=X_test_final.index)

errors_df = analysis_df[analysis_df['true_label'] != analysis_df['predicted_label']].copy()

def error_type(row):
    if row['true_label'] == 1 and row['predicted_label'] == 0:
        return 'False Negative, actual 1; predicted 0'
    elif row['true_label'] == 0 and row['predicted_label'] == 1:
        return 'False Positive, actual 0; predicted 1'

errors_df['error_type'] = errors_df.apply(error_type, axis=1)

errors_df['severity'] = (errors_df['predicted_prob'] - best_threshold).abs()

errors_df = errors_df.join(X_test_final)

errors_df = errors_df.sort_values(by='severity', ascending=False)

print('==== ERROR SUMMARY===')
print(f'error per: {len(errors_df)/len(X_test_final):.2%}')

#errors_df.to_csv('error_analysis_with_index.csv', index=True)
errors_df.head(20)
