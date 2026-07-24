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
feature_cols= [#'datetime', 
  #'tamb',  'pamb' ,
  #'psamp',
  
  'pflow', 'tmod',  'CH4_rt', 'CH4_w',
       'CH4_ht', 'CH4_area', 'CH4_skew', 'CH4_start_time', 'CH4_end_time',
       'CH4_start_level', 'CH4_end_level', #'duration', 
       'is_air','is_std',
       #'previous_type_std', 'previous_type_air','next_type_std', 'next_type_air', 
       'last_std_CH4_ht', 'last_air_CH4_ht', 'last_std_CH4_area',
       'last_air_CH4_area', 'last_std_CH4_rt', 'last_air_CH4_rt',
       'last_std_CH4_start_level', 'last_air_CH4_start_level',
       'level_rt_ratio', 'rt_position', 'baseline_slope', 'level_area_ratio'
       #'is_ht_zero_and_C_Nan','is_normal_std', 
       # #'is_bad_std', 'is_protential_flagged_air'
       ]

feature_config ={
    'diff':{'cols': ['CH4_area', 
                    'CH4_ht','CH4_w' ,#'tmod',#'CH4_end_time','CH4_start_time',,'CH4_rt'
                ], 'periods':[1]},

   # 'lag':{'cols':['CH4_area', 'CH4_rt',  'CH4_ht', #'tmod',  'psamp','pamb','tamb',,'CH4_skew'
                #'pflow',
      #         'CH4_w'], 'periods':[1]},

    

    'roll_std':{'cols': ['CH4_w', 'CH4_end_time','CH4_ht','CH4_area'], 'period':['40min','2h','6h']},

    'roll_mean_percent_res':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area'], 'period': ['40min','2h','6h']},

    'diff_cross':{'cols':[['CH4_end_time', 'CH4_start_time'],
                          # ['CH4_w_roll_mean_2h', 'CH4_w_roll_mean_7h'],
                           ['last_std_CH4_ht', 'last_air_CH4_ht'],#['CH4_end_level', 'CH4_start_level'],
                           ['CH4_ht', 'last_std_CH4_ht'], ['CH4_ht', 'last_air_CH4_ht'],#['CH4_ht', 'CH4_ht_lag_72'],['CH4_w', 'CH4_w_lag_72'],
                           #['pflow', 'pflow_lag_1'],['CH4_skew','CH4_skew_lag_1']
                           ]}, 

    'log':{'cols':['last_std_CH4_ht','last_air_CH4_ht'
                   #'CH4_end_time', 
                   ]},

    'multi':{'cols':[['CH4_w', 'CH4_ht'],['CH4_end_time_CH4_start_time_diff_cross','last_std_CH4_ht_log'], ['CH4_end_time_CH4_start_time_diff_cross','last_air_CH4_ht_log']]},

    'ratio':{'cols':[   
                        ['CH4_area','CH4_w_CH4_ht_multi'], ['CH4_ht', 'CH4_area'],
                        ['CH4_w', 'CH4_end_time_CH4_start_time_diff_cross'],
                       # ['CH4_rt', 'CH4_end_time_CH4_start_time_diff_cross'],

                        ['CH4_ht', 'last_air_CH4_ht'],# ['CH4_ht', 'last_air_ht'],
                        #['CH4_w_roll_std_2h', 'CH4_w_roll_std_1D'],
                        ['last_std_CH4_ht', 'last_air_CH4_ht'],
                        ['CH4_start_level', 'CH4_end_level'],
                        ['CH4_area', 'last_std_CH4_area'],['CH4_start_level', 'last_std_CH4_start_level'],
                        ['CH4_w_roll_std_40min', 'CH4_w_roll_mean_40min'], ['CH4_ht_roll_std_40min', 'CH4_ht_roll_mean_40min'],['CH4_area_roll_std_40min', 'CH4_area_roll_mean_40min'],
                        ['CH4_w_roll_std_2h', 'CH4_w_roll_mean_2h'], ['CH4_ht_roll_std_2h', 'CH4_ht_roll_mean_2h'],['CH4_area_roll_std_2h', 'CH4_area_roll_mean_2h'],
                        ['CH4_w_roll_std_6h', 'CH4_w_roll_mean_6h'], ['CH4_ht_roll_std_6h', 'CH4_ht_roll_mean_6h'],['CH4_area_roll_std_6h', 'CH4_area_roll_mean_6h'],
                        #['CH4_ht_roll_std_24h', 'CH4_ht_roll_mean_24h']
                       # ['CH4_w_roll_std_7h', 'CH4_w_roll_mean_7h'],['CH4_ht_roll_std_7h', 'CH4_ht_roll_mean_7h'],
                        #['CH4_w_roll_std_2D', 'CH4_w_roll_mean_2D'],['CH4_ht_roll_std_2D', 'CH4_ht_roll_mean_2D'],
                        #['CH4_rt', 'last_std_rt'],
                        
                       #['CH4_end_time', 'CH4_start_time'], ['CH4_area', 'pflow'],  ['CH4_skew', 'CH4_w'],['CH4_w', 'CH4_ht']
                       ]}, 


    #'per_change':{'cols':[['CH4_ht_diff_1', 'CH4_ht_lag_1'],['last_std_ht', 'last_air_ht']]},
    #'relative_per': {'cols': ['CH4_w', 'CH4_end_time','CH4_ht'], 'period':['2h','7h']},
    #'Z_score_res':{'cols':[['CH4_w', 'CH4_w_roll_mean_2h', 'CH4_w_roll_std_2h', '2h'], #['CH4_end_time', 'CH4_end_time_roll_mean_7D', 'CH4_end_time_roll_std_7D', '7D']
                           #]},
    #'per_rank': {'cols': ['CH4_w', 'CH4_skew','CH4_ht'], 'period':['7h','7D','30D']},


}
#===============

def get_data_by_year(file_path, year_list):
    df = pd.read_parquet(file_path, filters=[('year', 'in', year_list)]).set_index('datetime')
    return df


parquet_filename ='../data/processed/mhd_ch4_cnan_v1.parquet'
df = pd.read_csv('../data/processed/mhd_ch4_cnan_v1.csv', index_col= 'datetime')
df = df.drop(df[df['year']==2026].index)
def add_columns(df):
   

    target_cols = ['ht', 'area', 'rt', 'start_level']

    for col in target_cols:
        col = f'CH4_{col}'

        for t_type in ['std', 'air']:
            type_median = df.loc[df['type'] == t_type, col ].median()

            only_series = df[col].where(df['type']==t_type)
            df[f'last_{t_type}_{col}'] = only_series.ffill().shift(1).fillna(type_median)


    df['level_rt_ratio'] = (df['CH4_end_time'] -df['CH4_start_time'])/df['CH4_rt']
    df['rt_position'] = (df['CH4_rt'] - df['CH4_start_time']) / df['CH4_w']
    df['baseline_slope'] =(df['CH4_end_level'] -df['CH4_start_level'])/df['CH4_w']
    df['level_area_ratio'] = np.maximum(df['CH4_end_level'],df['CH4_start_level'] )/(df['CH4_area'])
    return df
    #===========

#==========================
years = df['year'].unique()
print(years)
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)
#===
#start_year = years.min()
#end_year = years.max() # not use the latest year
print("Years begin from", years.min(), "to", years.max())
#===== model
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

#======
pipe_rnd = Pipeline([
    #('scl', StandardScaler()),
    ('clf', RandomForestClassifier())
])
param_grid = {
    'clf__n_estimators': [100],
    'clf__max_depth': [5, 8],
    'clf__min_samples_leaf':[15],
    'clf__random_state':[42],
    'clf__max_features':['sqrt'],
    'clf__n_jobs':[4],
    'clf__class_weight':['balanced']
}

window = 1
test_size = 1
all_results = []
importance_dict = {}
test_importance_dict = {}

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
        y_train = data_train['label_c_nan']
        X_test = data_test[feature_cols]
        y_test = data_test['label_c_nan']
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
            train_size = 20000,
            stratify=y_train,
            random_state=42
        )
        grid = GridSearchCV(estimator=pipe_rnd, param_grid = param_grid,  scoring='average_precision', n_jobs=1, return_train_score=False, refit=True)
        grid.fit(X_search, y_search)

        del X_search, y_search
        gc.collect()

        best_clf = grid.best_estimator_
        best_clf.fit(X_train_final, y_train)

        importance_dict[train_years[0]] = pd.Series(best_clf.named_steps['clf'].feature_importances_, index=X_train_final.columns)

        perm_result = permutation_importance(
            estimator=best_clf,
            X=X_test_final,
            y=y_test,
            scoring = 'average_precision',
            n_repeats=5,
            random_state=42
        )
        test_importance_dict[test_years[0]] = pd.Series(
            perm_result.importances_mean,
            index=X_test_final.columns
        )
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
df_imp.to_csv('df_imp_mhd_1year_1year.csv')

df_imp_test = pd.DataFrame(test_importance_dict)
df_imp_test.to_csv('df_imp_test_mhd_1year_1year.csv')

print("============ Final Results=======")
print(rolling_results)
rolling_results.to_csv('mhd_rolling_results_all_feature_1year_1year.csv')

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