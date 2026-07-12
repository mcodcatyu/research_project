import numpy as np
import pandas as pd
import seaborn as sns
import gc
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

parquet_filename ='../data/processed/tac_co2_processed_v1.parquet'
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)
#mlflow.sklearn.autolog()

mlflow.set_experiment('rolling_windwo_tac_co2_optical')

def get_data_by_year(file_path, year_list):
    df = pd.read_parquet(file_path, filters=[('year', 'in', year_list)])
    return df

def get_data_by_year(file_path, year_list):
    df = pd.read_parquet(file_path, filters=[('year', 'in', year_list)])
    return df


class TSFE(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols,  feature_config, target_col='label1'):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.feature_config = feature_config
    #==========
    def fit (self, X, y=None):
        return self
    #=============
    def _fill_Nan(self, df, feature_col):
        for i in feature_col:
            df[i] = df[i].ffill().bfill()
        return df
    # ============
    def _ratio_featutre_gen(self, df):
        df['warmbox_temp_cavity_temp_ratio'] = df['warmbox_temp']/df['cavity_temp']

        fill_feature_ratio = ['warmbox_temp_cavity_temp_ratio']
        for i in fill_feature_ratio :
            df[i] = df[i].fillna(0)
        return df

    #=============== diff, lag, rolling
    def _diff_gen(self, df, feature, period):
        df[f'{feature}_diff_{period}'] = (df[f'{feature}'].diff(period)).fillna(0)

        return df
    #===============
    def _lag_gen (self, df, feature, period):
        df[f'{feature}_lag_{period}'] = (df[f'{feature}'].shift(period)).fillna(0)
        return df

    #===============
    def _rolling_std_gen (self, df, feature, period):
        df[f'{feature}_roll_std_{period}'] = (df[f'{feature}'].rolling(window=pd.to_timedelta(period), closed='left').std()).fillna(0)# self not included, NAN->0
        return df

    #===============
    def _rolling_mean_residual_gen(self, df, feature, period):
        df[f'{feature}_roll_mean_{period}'] = df[f'{feature}'].rolling(window=pd.to_timedelta(period), closed='left').mean().fillna(0) # self not included
        df[f'{feature}_residual_{period}'] = df[f'{feature}']- df[f'{feature}_roll_mean_{period}']# self not included
        return df
    #================
    def _feature_eng_apply(self, df, config):

        for opt, params in config.items():
            cols = params['cols']
            period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1

            for feature in cols:
                if opt == 'diff':
                    df = self._diff_gen(df, feature, period)
                elif opt == 'lag':
                    df = self._lag_gen(df, feature, period)
                elif opt == 'roll_std':
                    df = self._rolling_std_gen( df, feature, period)
                elif opt == 'roll_mean_res':
                    df = self._rolling_mean_residual_gen(df, feature, period)
        return df 

    def transform(self, X):
        df = X.copy()
        #df = self._fill_Nan(df, self.feature_cols)
        df = self._fill_Nan(df, self.feature_cols)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = self._ratio_featutre_gen(df)
        df = self._feature_eng_apply(df, self.feature_config)

        return df
    
def best_threshold(X_test, model, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    f1_scores = 2 * (precisions *recalls) / (precisions + recalls+ 1e-7) # 1e-7 to prevent Nan

    best_idx = np.argmax(f1_scores[:-1])
    best_thres = thresholds[best_idx]
    return best_thres, f1_scores[best_idx], precisions[best_idx], recalls[best_idx]
    
def random_forest_model(X_train, y_train, X_test, y_test, feature_names, 
                        experiment_run_name='RF_Model',n_estimators=200, 
                        max_depth=12, n_jobs=-1, threshold=0.5, class_weight=None, testyear=2012):
    """
    Fits and evaluate ML models.
    models: dictionary of different ML models
    X_train: training data (no labels)
    X_test: testinh data (no labels)
    y_train: training labels
    y_test: testing labels
    returns model scores dictionary
    """

    model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth = max_depth,
    max_samples=0.4,
    random_state=42,
    n_jobs = n_jobs,
    class_weight=class_weight)


    model.fit(X_train, y_train)

    # evaluation and recording Metrics
    y_proba = model.predict_proba(X_test)[:, 1]
    #y_pred = model.predict(X_test)
    y_pred = (y_proba > threshold).astype(int) # threshold should be 0.05 or something
    
    acc = accuracy_score(y_test, y_pred)

    print("Model training completed!")
    print(classification_report(y_test, y_pred))

    cm =confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax =plt.subplots(figsize=(6,6))
    disp.plot(ax=ax,cmap=plt.cm.Blues)
    plt.title(f'Normalized Confusion Matrix, threshold:{testyear}')
    # upload figure to mlflow
    plot_path = f'image/0711_confusion_matrix_{testyear}.png'
     # to produce the real confusion matrix plot to cover the mlflow autolog's(threshold=0.5 default) 
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()
    return model

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
    'roll_mean_res':{'cols':['cycle_time'], 'period': '5min'}
}


years_df = pd.read_parquet(parquet_filename, columns=['year'])
years = years_df['year'].unique()
#===
start_year = years.min()
end_year = years.max()
print("Years begin from", start_year, "to", end_year)
#===== model
tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

#========
def training(train_years,test_years,  start_year, all_results):
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

        model = random_forest_model(
        X_train_final, y_train, X_test_final, y_test,
        feature_names=feature_cols,
        experiment_run_name=f'rolling_{start_year}_{start_year+4}', n_estimators=100, max_depth=8, n_jobs=4,
        threshold=0.5,class_weight='balanced',testyear=test_years
        )
#====== 
        best_thres, best_f1, best_prec, best_rec = best_threshold(X_test_final, model, y_test)

        mlflow.log_param("training years", f"{start_year}-{start_year+4}")
        mlflow.log_param("test_year", test_years)
        mlflow.log_param("train_size", len(data_train))
        mlflow.log_param("test_size", len(data_test))

        mlflow.log_metric("Best threshold", best_thres)
        mlflow.log_metric("Best F1-score",best_f1)
        mlflow.log_metric("Precision", best_prec)
        mlflow.log_metric("Recall", best_rec)
        all_results.append({
            "training_years": f"{start_year}_{start_year+4}",
            "testing year":test_years,
            "threshold":best_thres,
            "F1-score":best_f1,
            "Precision": best_prec,
            "Recall":best_rec,
            "train_size":len(data_train),
            "test_size":len(data_test)
        })
        del data_train, data_test,  X_train_final, y_train, X_test_final, y_test,  X_train, X_test,model, best_thres, best_f1, best_prec, best_rec 
        gc.collect()
        return all_results
#==========
window = 3
all_results = []
with mlflow.start_run(run_name="0711_v2_window_3") as parent_run:
    for start_year in years:
        with mlflow.start_run(run_name=f'rol_{start_year}', nested=True):
            train_years = list(range(start_year, start_year + window))
            test_years = [start_year + window]
            print("train_years:", train_years)
            print("test_years:", test_years)
            if test_years not in years:
                print(f"{test_years} data not in the dataset")
                break
            all_results = training(train_years,test_years, start_year, all_results)
            
            #del data_train, data_test,  X_train_final, y_train, X_test_final, y_test,  X_train, X_test,model, best_thres, best_f1, best_prec, best_rec 
            #gc.collect()
    rolling_results = pd.DataFrame(all_results)
    print(rolling_results)




fig, ax = plt.subplots(figsize=(8,5))
sns.lineplot(data=rolling_results, x='testing year', y='F1-score',ax=ax, label='unbalanced', marker='o')
ax.set_title("F1-score Comparision unbalanced")
ax.set_ylabel('F1-score')
ax.set_xlabel('testing year')
plt.savefig('model/f1_score_0711_v1')
mlflow.log_artifact('model/f1_score_0711_v1')
plt.close()

