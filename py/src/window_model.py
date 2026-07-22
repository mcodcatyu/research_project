import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import TunedThresholdClassifierCV, TimeSeriesSplit, RandomizedSearchCV
import mlflow
from sklearn.base import BaseEstimator, TransformerMixin
#===================
parquet_filename ='../data/processed/tac_co2_processed_v1.parquet'
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)
mlflow.sklearn.autolog() # 自動記錄

mlflow.set_experiment('rolling_windwo_tac_co2_optical') # 實驗名稱

#=============
#這邊要是 TSFE
#===========

#========= Feature cols & config =========
feature_cols= [ 
         'co2_dry','co2_wet', 
        'co2_target_error',  #'co2_C',         'co2_N',     
        'co2_Cdrift', 'co2_Nfiltered', 
       'cycle_time', 'h2o', 'cavity_press', 'cavity_temp',
       'das_temp', 'etalon_temp', 'warmbox_temp', 'outlet_valve', 'datetime'
       #'label1'
]

feature_config ={
    'diff':{'cols': [ 'warmbox_temp','cavity_temp',
                     'das_temp','cycle_time'], 'periods':1},
    'lag':{'cols':['cycle_time'], 'periods':1},

    'roll_std':{'cols': ['cavity_press', 'cavity_temp',
       'das_temp', 'etalon_temp', 'warmbox_temp'], 'period':'5min'},
    'roll_mean_percent_res':{'cols':['cycle_time'], 'period': '5min'}
}
#=================
#每次只拿指定的年份的資料，避免記憶體爆炸

years_df = pd.read_parquet(parquet_filename, columns=['year'])
years = years_df['year'].unique() #取得年分

# 取得範圍
start_year = years.min() 
end_year = years.max()
print("Years begin from", start_year, "to", end_year)

tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

window = 5 # window大小
all_results = []

with mlflow.start_run(run_name="0711_v1") as parent_run:
    for start_year in years:
        train_years = list(range(start_year, start_year + window))
        test_years = [start_year + window]
        print("train_years:", train_years)
        print("test_years:", test_years)
        if test_years not in years:
            print(f"{test_years} data not in the dataset")
            break

        data_train = get_data_by_year(parquet_filename, train_years)
        data_test = get_data_by_year(parquet_filename, test_years)
        
        #data_train['datetime'] = pd.to_datetime(data_train['datetime'])
        #data_test['datetime'] = pd.to_datetime(data_test['datetime'])
        
        X_train = data_train[feature_cols]
        y_train = data_train['label1']
        X_test = data_test[feature_cols]
        y_test = data_test['label1']
        # feature engineering
        X_train_final = tsfe.fit_transform(X_train)
        X_test_final = tsfe.fit_transform(X_test)

        #
    
        y_train.index= X_train_final.index
        y_test.index =X_test_final.index
        
        #這邊改成 grid 看看那樣
        model = random_forest_model(
        X_train_final, y_train, X_test_final, y_test,
        feature_names=feature_cols,
        experiment_run_name=f'rolling_{start_year}_{start_year+4}', n_estimators=100, max_depth=8, n_jobs=-1,
        threshold=0.5,class_weight=None
        )
       #====== 
        best_thres, best_f1, best_prec, best_rec = best_threshold(X_test_final, model, y_test)
    
        mlflow.log_param("training years", f"{start_year}-{test_years-1}")
        mlflow.log_param("test_year", test_years)
        mlflow.log_param("train_size", len(data_train))
        mlflow.log_param("test_size", len(data_test))

        mlflow.log_metric("Best threshold", best_thres)
        mlflow.log_metric("Best F1-score",best_f1)
        mlflow.log_metric("Precision", best_prec)
        mlflow.log_metric("Recall", best_rec)

        all_results.append({
            "training_years": f"{start_year}_{end_year-1}",
            "testing year":test_years,
            "threshold":best_thres,
            "F1-score":best_f1,
            "Precision": best_rec,
            "Recall":best_rec,
            "train_size":len(data_train),
            "test_size":len(data_test)
        })
    rolling_results = pd.DataFrame(all_results)

