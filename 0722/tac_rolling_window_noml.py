import numpy as np
import pandas as pd
import seaborn as sns
import gc
import matplotlib.pyplot as plt
sns.set_theme(style='darkgrid')
pd.set_option('display.max_columns', None)

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

parquet_filename ='../data/processed/tac_co2_processed_v1.parquet'
#mlflow.set_tracking_uri(
 #   "http://127.0.0.1:5000"
#)
#mlflow.sklearn.autolog()

#mlflow.set_experiment('rolling_window_tac_co2_optical')


def get_data_by_year(file_path, year_list):
    df = pd.read_parquet(file_path, filters=[('year', 'in', year_list)]).set_index('datetime')
    return df

#=====
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

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

        df[f'{feature[0]}_{feature[1]}_per_change'] = (df[feature[0]]/df[feature[1]])*100
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
            df[f'{feature}_residual_{p}'] = ((df[f'{feature}']- df[f'{feature}_roll_mean_{p}'])/(df[f'{feature}_roll_mean_{p}']))*100 # self not included
        return df
    #================
    def _Zscore_res_gen(self, df, feature):
        df[f'{feature[0]}_{feature[3]}_zcore_res_gen'] = ((df[feature[0]]-df[feature[1]])/df[feature[2]])
        return df
    #=================
    def _log_gen (self, df, feature):
        df[f'{feature}_log'] = np.sign(df[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))#.fillna(df[f'{feature}_lag_{p}'].median())
        return df
    #===========
    def _relative_per_gen(self, df, feature, period):
        for p in period:
            df[f'{feature}_relative_per_{p}'] = (df[f'{feature}']-(df[f'{feature}'].rolling(window=p, closed='left').min()))/((df[f'{feature}'].rolling(window=p, closed='left').max())-(df[f'{feature}'].rolling(window=p, closed='left').min())) #.fillna(df[f'{feature}_roll_std_{p}'].median())# self not included, NAN->0
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

        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values

        df.index = pd.to_datetime(df.index)
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values
        df_columns = df.columns.tolist()
        df = self._fill_Nan(df, df_columns)

        floas_cols = df.select_dtypes(include=['float64']).columns
        df[floas_cols] = df[floas_cols].astype('float32')

        return df
    
#==========
from sklearn.metrics import precision_recall_curve
def get_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    p = precisions[:-1]
    r = recalls[:-1]

    f1_scores = 2 * (p * r) / (p + r + 1e-7) # 1e-7 to prevent Nan

    best_index = np.argmax(f1_scores)
    best_thres = thresholds[best_index]
    return best_thres, f1_scores[best_index], p[best_index], r[best_index]


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
def add_columns(df):
    switch_flag = df['port'].ne(df['port'].shift(1))|df['sample'].ne(df['sample'].shift(1))


    switch_group = switch_flag.cumsum()

    group_start_time = pd.Series(df.index, index=df.index).groupby(switch_group).transform('first')

    df['time_since_switch'] = (df.index -group_start_time ).dt.total_seconds()

    group_start_cycle = df['cycle_time'].groupby(switch_group).transform('first')

    df['cycle_time_diff'] = df['cycle_time'] - group_start_cycle

    return df

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
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)
#===
start_year = years.min()
end_year = years.max()-1 # not use the latest year
print("Years begin from", start_year, "to", end_year)
#===== model
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

#======
pipe_rnd = Pipeline([
    #('scl', StandardScaler()),
    ('clf', RandomForestClassifier())
])
param_grid = {
    'clf__n_estimators': [50],
    'clf__max_depth': [6],
    'clf__min_samples_leaf':[50],
    'clf__max_samples':[0.2, 0.5],
    'clf__random_state':[42],
    'clf__max_features':['sqrt'],
    'clf__n_jobs':[1],
    'clf__class_weight':['balanced']
}

window = 2
test_size = 1
all_results = []
importance_dict = {}

name = "0721_v1_window_1_all"

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
        data_train = add_columns(data_train)
        data_test = add_columns(data_test)

        #=========
        #data_train['datetime'] = pd.to_datetime(data_train['datetime'])
        #data_train = data_train.set_index('datetime').sort_index()
        #data_test['datetime'] = pd.to_datetime(data_test['datetime'])
    # data_test = data_test.set_index('datetime').sort_index()
        
        #data_train['datetime'] = pd.to_datetime(data_train['datetime'])
        #data_test['datetime'] = pd.to_datetime(data_test['datetime'])

        X_train = data_train[feature_cols]
        y_train = data_train['label1']
        X_test = data_test[feature_cols]
        y_test = data_test['label1']
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

        X_search, _, y_search, _ = train_test_split(
            X_train_final, y_train,
            train_size = 30000,
            stratify=y_train,
            random_state=42
        )
        grid = GridSearchCV(estimator=pipe_rnd, param_grid = param_grid,  scoring='average_precision', n_jobs=1, return_train_score=False, refit=True)
        grid.fit(X_search, y_search)

        del X_search, y_search
        gc.collect()

        best_clf = grid.best_estimator_
        best_clf.fit(X_train_final, y_train)

        importance_dict[test_years[0]] = pd.Series(best_clf.named_steps['clf'].feature_importances_, index=X_train_final.columns)

        #model = random_forest_model(
        #X_train_final, y_train, X_test_final, y_test,
        #feature_names=feature_cols,
    # experiment_run_name=f'rolling_{start_year}_{start_year+4}', n_estimators=100, max_depth=8, n_jobs=-1,
    # threshold=0.5,class_weight=None
    # )
    #====== 
        train_f1 = f1_score(y_train, best_clf.predict(X_train_final))
        test_f1 = f1_score(y_test, best_clf.predict(X_test_final))
        
        y_prob = best_clf.predict_proba(X_test_final)[:, 1]
        best_threshold, best_f1, best_prec, best_rec = get_best_threshold(y_test, y_prob)
        y_proba_best = (y_prob >= best_threshold).astype(int)

        test_pr_auc = average_precision_score(y_test, y_prob)


        print(f'Test PR-AUC: {test_pr_auc:.4f}')
        print(f'Train / Test F1 Gap : {train_f1 - test_f1:.4f}')


        all_results.append({
            "Training_years": f"{start_year}_{start_year+window-1}",
            "Testing year":test_years,
            'Best Params': str(grid.best_params_),
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
        del  data_train, data_test, X_train_final, X_test_final, y_train, y_test, y_prob, best_clf, grid
        gc.collect()
rolling_results = pd.DataFrame(all_results)

df_imp = pd.DataFrame(importance_dict)
df_imp.to_csv('df_imp.csv')

print("============ Final Results=======")
print(rolling_results)
rolling_results.to_csv('rolling_results_all_feature.csv')

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