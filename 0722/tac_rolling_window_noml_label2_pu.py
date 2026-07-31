import numpy as np
import pandas as pd
import seaborn as sns
import gc
import matplotlib.pyplot as plt
sns.set_theme(style='darkgrid')
pd.set_option('display.max_columns', None)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,average_precision_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import TunedThresholdClassifierCV, TimeSeriesSplit, RandomizedSearchCV
import mlflow
from sklearn.base import BaseEstimator, TransformerMixin
from feature_eng import TSFE
from function import get_best_threshold, get_data_by_year, optiacal_add_columns

from pulearn import ElkanotoPuClassifier
from sklearn.ensemble import RandomForestClassifier

#======================
parquet_filename ="../data/processed/tac_co2_formatted_v2_label2.parquet"
#mlflow.set_tracking_uri(
 #   "http://127.0.0.1:5000"
#)
#mlflow.sklearn.autolog()

#mlflow.set_experiment('rolling_window_tac_co2_optical')



#========   
feature_cols= [ 
         'co2_dry','co2_wet', 
        'co2_target_error',  #'co2_C',         'co2_N',     
        'co2_Cdrift', 'co2_Nfiltered', 
       'cycle_time', 'h2o', 'cavity_press', 'cavity_temp',
       'das_temp', 'etalon_temp', 'warmbox_temp', 'outlet_valve', #'datetime'
       'cycle_time_diff','time_since_switch'
]

feature_config ={
    'diff':{'cols': [ 'warmbox_temp','cavity_temp',
                     'das_temp','cycle_time','cavity_press',
                     'h2o','co2_dry'], 'periods':[1, 2]},

    'lag':{'cols':['cycle_time'], 'periods':[1,2]},

    'roll_std':{'cols': ['cavity_press', 'cavity_temp',
            'das_temp', 'etalon_temp', 'warmbox_temp', 'co2_target_error','cycle_time'], 'period':['60s', '5min']},

    'roll_mean_percent_res':{'cols':['cycle_time','co2_target_error'], 'period': ['60s','5min']},

    'diff_cross':{'cols':[['co2_wet','co2_dry'],
                           ]}, 

    'log':{'cols':['h2o','co2_target_error'
                   #'CH4_end_time', 
                   ]},

    'multi':{'cols':[['h2o', 'co2_dry'],['cavity_press', 'cavity_temp'], ['outlet_valve','cavity_press_diff_1']]},

    'ratio':{'cols':[   
                        ['outlet_valve','cavity_press'], 
                        
                        
                        ['cycle_time_roll_std_5min', 'cycle_time_roll_mean_5min'], 
                        ['cavity_press_roll_std_5min', 'cavity_temp_roll_std_5min'],
                        ['cavity_press_roll_std_60s', 'cavity_temp_roll_std_60s'],
                       ]}, 


    #'per_change':{'cols':[['CH4_ht_diff_1', 'CH4_ht_lag_1'],['last_std_ht', 'last_air_ht']]},
    #'relative_per': {'cols': ['CH4_w', 'CH4_end_time','CH4_ht'], 'period':['2h','7h']},
    'Z_score_res':{'cols':[['co2_target_error', 'co2_target_error_roll_mean_5min', 'co2_target_error_roll_std_5min', '5min'], #['CH4_end_time', 'CH4_end_time_roll_mean_7D', 'CH4_end_time_roll_std_7D', '7D']
                           ]},
    'per_rank': {'cols': ['co2_target_error', 'co2_Cdrift'], 'period':['60s','5min']},


}
#===============

#===========
feature_ml = ['time_since_switch',
 'co2_target_error_per_rank_60s',
 'cycle_time_residual_5min',
 'cycle_time_roll_std_5min_cycle_time_roll_mean_5min_ratio',
 'cycle_time_roll_std_5min',
 'co2_Cdrift_per_rank_60s',
 'cycle_time_residual_60s',
 'etalon_temp',
 'cycle_time',
 'cycle_time_lag_1',
 'cycle_time_lag_2',
 'outlet_valve',
 'outlet_valve_cavity_press_ratio',
 'cycle_time_roll_mean_60s',
 'cavity_press_roll_std_5min_cavity_temp_roll_std_5min_ratio',
 'co2_Nfiltered',
 'cycle_time_diff',
 'h2o_log',
 'co2_wet_co2_dry_diff_cross',
 'h2o',
 'h2o_co2_dry_multi',
 'co2_target_error_residual_5min',
 #'cycle_time_diff_2',
 #'co2_target_error_per_rank_5min',
 #'cavity_temp_roll_std_5min'
]
#==========================


years_df = pd.read_parquet(parquet_filename, columns=['year'])
years = years_df['year'].unique()
years =years_df.loc[~years_df['year'].isin([2025,2012]), 'year'].unique()
print(years)
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)
#===
#start_year = years.min()
#end_year = years.max() # not use the latest year
print("Years begin from", years.min(), "to", years.max())
#===== model
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

#======
#pipe_rnd = Pipeline([
   # #('scl', StandardScaler()),
   # ('clf', RandomForestClassifier())
#])
#param_grid = {
 #   'clf__n_estimators': [50],
 #   'clf__max_depth': [6],
  #  'clf__min_samples_leaf':[50],
 #   'clf__max_samples':[0.2, 0.5],
  #  'clf__random_state':[42],
 #   'clf__max_features':['sqrt'],
 #   'clf__n_jobs':[1],
 #  'clf__class_weight':['balanced']
#}


window = 2
test_size = 1
all_results = []
importance_dict = {}
test_importance_dict = {}


name = "0730_v1_window_1_all"


for start_year in years:
        train_years = list(range(start_year, start_year + window))
        test_years = list(range(start_year + window, start_year + window+ test_size))
        print("train_years:", train_years)
        print("test_years:", test_years)
        if test_years not in years:
            print(f"{test_years} data not in the dataset")
            break

        data_train = get_data_by_year(parquet_filename, train_years)
        data_test = get_data_by_year(parquet_filename, test_years)
        data_train = optiacal_add_columns(data_train)
        data_test = optiacal_add_columns(data_test)

        #=========
        #data_train['datetime'] = pd.to_datetime(data_train['datetime'])
        #data_train = data_train.set_index('datetime').sort_index()
        #data_test['datetime'] = pd.to_datetime(data_test['datetime'])
    # data_test = data_test.set_index('datetime').sort_index()
        
        #data_train['datetime'] = pd.to_datetime(data_train['datetime'])
        #data_test['datetime'] = pd.to_datetime(data_test['datetime'])

        X_train = data_train[feature_cols]
        y_train = data_train['label2']
        X_test = data_test[feature_cols]
        y_test = data_test['label2']
        # feature engineering
        X_train = tsfe.fit_transform(X_train)
        X_test = tsfe.transform(X_test)

        #
        X_train_final = X_train #[feature_ml]
        X_test_final = X_test #[feature_ml]
        y_train.index= X_train_final.index
        y_test.index =X_test_final.index
        
        del X_train, X_test, 
        gc.collect()
    #====================== PU Learning model
        base_rf = RandomForestClassifier(
              n_estimators=50,
              max_depth=6,
              min_samples_leaf=5,
              random_state=42
        )

        pu_estimator = ElkanotoPuClassifier(
              estimator=base_rf,
              hold_out_ratio=0.2
        )

        pu_estimator.fit(X_train_final, y_train)

        #===========
        underlying_rf = pu_estimator.estimator
        importance_dict[train_years[0]] = pd.Series(
            underlying_rf.feature_importances_,
            index=X_train_final.columns
        )

        perm_result = permutation_importance(
              estimator = pu_estimator,
              X=X_test_final,
              y=y_test,
              scoring = 'average_precision',
              n_repeats=5,
              random_state=42
        )

        #===================================
        train_f1 = f1_score(y_train, pu_estimator.predict(X_train_final))
        test_f1 = f1_score(y_test, pu_estimator.predict(X_test_final))

        y_prob = pu_estimator.predict_proba(X_test_final)[:, 1]
        best_threshold, best_f1, best_prec, best_rec = get_best_threshold(y_test, y_prob)

        test_pr_auc = average_precision_score(y_test, y_prob)

        print(f'Test PR-AUC: {test_pr_auc:.4f}')
        print(f'Train / Test F1 Gap : {train_f1 - test_f1:.4f}')

        all_results.append({
            "Training_years": f"{start_year}_{start_year+window-1}",
            "Testing year":test_years,
            'PR-AUC': round(test_pr_auc, 4),
            "Best threshold":best_threshold,
            "F1-score_test":best_f1,
            "Precision_test": best_prec,
            "Recall_test":best_rec,
            'F1 Gap (Train-Test)': train_f1 - test_f1,
            "Train_size":len(data_train),
            "Test_size":len(data_test)
        })
        start_year += window
        #import os, psutil
        #process = psutil.Process(os.getpid())
        #print(f"Memory Usage: {process.memory_info().rss /1024/1024:.2f }MB")
        del  data_train, data_test, X_train_final, X_test_final, y_train, y_test, y_prob
        gc.collect()

rolling_results = pd.DataFrame(all_results)

df_imp = pd.DataFrame(importance_dict)
df_imp.to_csv('df_imp_0731_label2_pu.csv')

df_imp_test = pd.DataFrame(test_importance_dict)
df_imp_test.to_csv('df_imp_test_0731_label2_pu.csv')

print("============ Final Results=======")
print(rolling_results)
rolling_results.to_csv('rolling_results_all_feature_2year_1year_0731_label2_pu.csv')

#metric_cols = ['PR-AUC',"F1-score_test", "Precision_test", "Recall_test" , 'F1 Gap (Train-Test)']
#summary_row = {"Training_Year":"Mean +- Std", "Test Year": "-", "Best Params": "-"}

#for col in metric_cols:
#    mean_val = rolling_results[col].mean()
#    std_val = rolling_results[col].std()
#    summary_row[col]

#summary_df = pd.DataFrame([summary_row])
#final_paper_table = pd.concat([rolling_results, summary_df], ignore_index=True)


#print(final_paper_table.to_string(index=False))
#final_paper_table.to_csv('final_paper_table_2_1.csv')