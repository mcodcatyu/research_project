import pandas as pd
import io
import os
import tempfile
import numpy as np
from pulearn import ElkanotoPuClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import datetime
import joblib
#可以把這邊改成classs，然後一進來就直接根據他選的地方去做預測
#===========================
class GCMDprocessor:
    def __init__ (self, uploaded_file, type_col='type'):
        self.uploaded_file = uploaded_file
        self.df=None
        self.type_col = type_col
        self.feature_cols= [
                    'pflow', 'tmod',  'rt', 'w','type',
                    'ht', 'area', 'skew', 'start_time', 'end_time',
                    'start_level', 'end_level', 
                ]
        self.feature_ml = ['rt_to_last_air_ratio',
                        'ht_roll_std_24h_ht_roll_mean_24h_ratio',
                        'area_roll_std_24h_area_roll_mean_24h_ratio',
                        'end_time',
                        'w_roll_std_24h_w_roll_mean_24h_ratio',
                        'ht_to_last_std_ratio',
                        'area_roll_std_3h_area_roll_mean_3h_ratio',
                        'w_residual_6h',
                        'ht_roll_std_3h_ht_roll_mean_3h_ratio',
                        'rt_pflow_ratio',
                        'level_area_ratio',
                        'area_to_last_air_ratio',
                        'w_residual_3h',
                        'w_robust_residual_3h',
                        'w_robust_residual_6h',
                        'area_robust_residual_1h',
                        'area_residual_1h',
                        'area_to_last_std_ratio',
                        'start_level_to_last_std_ratio',
                        'w_diff_1',
                        'w_roll_std_3h_w_roll_mean_3h_ratio',
                        'area_diff_1',
                        'w_residual_24h',
                        'ht_pflow_ratio',
                        'rt_to_last_std_ratio',
                        'start_time',
                        'ht_residual_1h',
                        'w',
                        'ht_diff_1',
                        'w_residual_1h']
        self.feature_config ={
                    'diff':{'cols': ['area', 
                                    'ht','w' , 'rt' 
                                ], 'periods':[1, 3]},
        
                    'lag':{'cols':['area', 'rt',  'ht', 
                                'pflow','w'], 
                                'periods':[1, 3]},
        
                    'diff_cross':{'cols':[
                                        ['end_time', 'start_time'],
                                        ['end_level', 'start_level'],
                                        ['rt', 'pflow'],
                                        ]}, 
        
                    'roll_std':{'cols': ['w', 'ht','area', 'end_time_start_time_diff_cross'], 
                                'period':['1h','3h','6h', '24h', '3D', '7D']},
        
                    'roll_mean_percent_res':{'cols':['rt' ,
                                            'w', 'ht','area',], 'period': ['1h','3h','6h','24h', '3D', '7D']},
        
        
        
                    #'log':{'cols':['last_std_ht','last_air_ht'
                                #'end_time', 
                    #              ]},
                    'roll_median':{'cols':['rt' ,
                                            'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h', '3D', '7D']},
                    'roll_mad':{'cols':['rt' ,
                                            'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h', '3D', '7D']},
                    'roll_median_percent_res':{'cols':['rt' ,
                                            'w', 'ht','area','pflow'],
                                            'period':['1h','3h','6h', '24h', '3D', '7D']},
        
                # 'multi':{'cols':[['w', 'ht'],['area', 'pflow']]},
        
                    'ratio':{'cols':[   ['skew','w'],
                                        ['ht', 'pflow'],
                                        ['ht', 'w'],
                                        ['area', 'pflow'],
                                        ['rt', 'pflow'],
                                        ['ht', 'area'],
                                        ['w', 'end_time_start_time_diff_cross'],
        
                                        ['start_level', 'end_level'],
                                
                                        ['w_roll_std_3h', 'w_roll_mean_3h'], ['ht_roll_std_3h', 'ht_roll_mean_3h'],['area_roll_std_3h', 'area_roll_mean_3h'],
                                        ['w_roll_std_24h', 'w_roll_mean_24h'], ['ht_roll_std_24h', 'ht_roll_mean_24h'],['area_roll_std_24h', 'area_roll_mean_24h'],
                                    ]}, 
        
        
                    'per_change':{'cols':[['ht_diff_1', 'ht_lag_1']]},
                    'relative_per': {'cols': ['w', 'ht'], 'period':['2h','7h']},
                    'Z_score_res':{'cols':[['w', 'w_roll_mean_3h', 'w_roll_std_3h', '3h'], 
                                        ]},
                    'per_rank': {'cols': ['w', 'skew','ht'], 'period':['7D']},
                    }
        

    def _fix_feature_names(self, names):
        unique_names=[]
        ht_seen=False

        for name in names:
            if name == 'ht' and not ht_seen:
                unique_names.append('inlet_ht')
                ht_seen=True
            else:
                unique_names.append(name)

        return unique_names

    def _parse_file(self):
        if isinstance(self.uploaded_file, str):
            file_path = self.uploaded_file
            cleanup_temp = False
        else:
            suffix = os.path.splitext(self.uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp_file:
                tmp_file.write(self.uploaded_file.getvalue())
                file_path = tmp_file.name
            cleanup_temp = True

        try:
            flag_total_len = 28
            with open(file_path, 'r', encoding='utf-8') as f:
            #skip header
                lines = f.readlines()
            header_line = lines[2].rstrip('\r\n')

            header_data_part = header_line[:-flag_total_len ]
            header_flag_part = header_line[-flag_total_len:]

            data_headers = header_data_part.strip().split()
            flag_headers = [
                header_flag_part[0:8].strip(),
                header_flag_part[8:16].strip(),
                header_flag_part[16:21].strip(),
                header_flag_part[21:28].strip()
            ] 

            raw_feature_names = data_headers + flag_headers
            feature_names = self._fix_feature_names(raw_feature_names)

            df_mhd=pd.read_fwf(
            file_path,
            skiprows=3,
            header=None,
            names=feature_names,
            keep_default_na=False
            )

            df_mhd.columns = feature_names

            # read the file in the way that the flag space is right

            df_mhd_data = pd.read_csv(
            file_path, 
            skiprows=3,
            sep=r'\s+',header=None, 
            names = feature_names,
            engine='python') # skip first row, txt is space seperated
            #這邊是為了 把flag和真實的data放一起，所以各只取flag 和flag以外的數值
            
            df_mhd_data.columns = feature_names

            real_data_mhd = df_mhd_data.iloc[:, :-4]

            df_mhd_flag = df_mhd.iloc[:, -4:]

            df_mhd_flag.index = real_data_mhd.index 

            real_data_mhd = pd.concat([real_data_mhd, df_mhd_flag], axis=1)
            real_data_mhd[['flag_ht','flag_a',  'flag', 'flag_p']] = real_data_mhd[['flag_ht','flag_a',  'flag', 'flag_p']].fillna(" ")
            mapping={" ":0, 
                "x":1, 
                "*":1,
                "F":1, 
                "B":0, 
                "A":0
            }
            real_data_mhd ["flag_ht_encod"] = real_data_mhd ["flag_ht"].map(mapping)
            real_data_mhd ["flag_a_encod"] = real_data_mhd ["flag_a"].map(mapping)
            real_data_mhd ["flag_label"] = ((real_data_mhd ["flag_ht_encod"]==1) | (real_data_mhd ["flag_a_encod"]==1)).astype(int)
            real_data_mhd=real_data_mhd.drop(columns=["flag_a_encod", "flag_ht_encod"])
            self.df = self._process_datetime(real_data_mhd)
        #感覺這邊應該順便作時間處裡? 設datetime為index
            return self.df
        finally:
            if cleanup_temp and os.path.exists(file_path):
                os.remove(file_path)



    def _process_datetime(self, df):
        df['date'] = df['date'].astype(str).str.zfill(6)
        df['time'] = df['time'].astype(str).str.zfill(6)
                
                
        df['datetime_str'] = df['date'] + df['time']

        df['datetime'] = pd.to_datetime(df['datetime_str'], format='%y%m%d%H%M%S')

        df.set_index('datetime', inplace=True)
        df.drop(columns=['datetime_str'], inplace=True, errors='ignore')
        return df 
    
    def _preprocessing(self, df):
        target = 'flag_label'        
        #預測的代碼放這
        X = df[self.feature_cols]
        y = df[target]
        split_idx = int(len(df)*0.8)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        tsfe = TSFE(feature_cols=self.feature_cols, feature_config=self.feature_config)

        X_train_final = tsfe.transform(X_train)
        X_test_final = tsfe.transform(X_test)
        X_train_final = X_train_final[self.feature_ml]
        X_test_final = X_test_final[self.feature_ml]
        #=================
        return X_train_final, y_train, X_test_final, y_test

    def _predict(self, df):
        X_pred = df[self.feature_cols]
        tsfe = TSFE(feature_cols=self.feature_cols, feature_config=self.feature_config)

        X_pred_final = tsfe.transform(X_pred)
        X_pred_final = X_pred_final[self.feature_ml]
        return X_pred_final

#============================== TSFE ========================================================
class TSFE:
    def __init__(self, feature_cols,  feature_config, type_col='type'):
        """
            class TSFE 入口
            Args:
                feature_cols(list):選擇使用的原始特徵欄位
                feature_config(dict): 轉換的特徵類別以及要產生的特徵
                type_col(string): 樣本類型
        """
        self.feature_cols = feature_cols
        self.feature_config = feature_config
        self.type_col = type_col
    #==========
    def _fill_Nan(self, df, feature_cols):
        fill_cols = [c for c in feature_cols if c!='type']
        df[fill_cols] = df[fill_cols].ffill().fillna(df[fill_cols].median()) # df[feature_cols].median()
        return df
    #===================
    def _basic_feature(self, df):
        #print("df passed 1 step!")

        #========
        target_cols = ['ht', 'area', 'rt', 'start_level']

        for col in target_cols:
            for t_type in ['std', 'air']:
                type_median = df.loc[df['type'] == t_type, col].median()

                only_series = df[col].where(df['type']==t_type)
                df[f'last_{t_type}_{col}'] = only_series.ffill().shift(1).fillna(type_median)

        for col in ['ht', 'area', 'rt', 'start_level']:
            df[f'{col}_to_last_std_ratio'] = df[col] / (df[f'last_std_{col}'] + 1e-7)
            df[f'{col}_to_last_air_ratio'] = df[col] / (df[f'last_air_{col}'] + 1e-7)

        df['duration_rt_ratio'] = (df['end_time'] -df['start_time'])/df['rt']
        df['rt_position'] = (df['rt'] - df['start_time']) / df['w']
        df['baseline_slope'] =(df['end_level'] -df['start_level'])/df['w']
        df['level_area_ratio'] = np.maximum(df['end_level'],df['start_level'] )/(df['area']+1e-7)

        return df
    
    #===============
    def _gen_single_feature(self, df, opt, feature, period=None):
        if self.type_col in df.columns:
            g = df.groupby(self.type_col)[feature]
        else:
            g = df[feature]

        def _clean(reset):
            if isinstance(reset.index, pd.MultiIndex):
                return reset.reset_index(0, drop=True).sort_index()
            return reset
        
        if opt == 'diff':
            for p in period:
                df[f'{feature}_diff_{p}'] = _clean(g.diff(p))

        elif opt == 'lag':
            for p in period:
                df[f'{feature}_lag_{p}'] = _clean(g.shift(p))
        elif opt == 'roll_std':
            for p in period:
                df[f'{feature}_roll_std_{p}'] = _clean(g.rolling(window=p, closed='left').std())
        elif opt == 'roll_mean_percent_res':
            for p in period:
                mean_col = _clean(g.rolling(window=p, closed='left').mean())
                df[f'{feature}_roll_mean_{p}'] =mean_col #.fillna(df[f'{feature}_roll_mean_{p}'].median()) # self not included
                df[f'{feature}_residual_{p}'] = ((df[f'{feature}']- mean_col)/(mean_col+1e-7))*100 # self not included

        elif opt =='log':
            df[f'{feature}_log'] = np.sign(df[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))
        elif opt == 'relative_per':
            for p in period:
                roll_min = _clean(g.rolling(window=p, closed='left').min())
                roll_max = _clean(g.rolling(window=p, closed='left').max())
                df[f'{feature}_relative_per_{p}'] = (df[f'{feature}']-(roll_min))/((roll_max)-(roll_min)+1e-7)
        elif opt == 'per_rank':
            for p in period:
                df[f'{feature}_per_rank_{p}'] = _clean(g.rolling(window=p, closed='left').rank(pct=True))
        elif opt == 'roll_median':
            for p in period:
                df[f'{feature}_roll_median_{p}'] = (
                    _clean(g.rolling(window=p, closed='left').median())
                )
        elif opt == 'roll_mad':
            def _calc_mad(x):
                med = np.median(x)
                return np.median(np.abs(x-med))

            for p in period:
                df[f'{feature}_mad_{p}']=(
                    _clean(g.rolling(window=p, closed='left').apply(_calc_mad, raw=True))
                )
                
        elif opt == 'roll_median_percent_res':
            """
            Robust residual ( percentage deviation of the current point from the median)
            """
            for p in period:
                median_col =df[f'{feature}_roll_median_{p}']
                df[f'{feature}_robust_residual_{p}'] = (
                    (df[f'{feature}']-median_col)/(median_col +1e-7 )*100
                )

    def _gen_cross_feature(self, df, opt, feat, period):
        f0, f1 = feat[0], feat[1]
        if opt == 'ratio':
            df[f'{f0}_{f1}_ratio'] = df[f0]/df[f1]
        elif opt == 'diff_cross':
            df[f'{f0}_{f1}_diff_cross'] = df[f0]-df[f1]
        elif opt == 'multi':
            df[f'{f0}_{f1}_multi'] = df[f0]*df[f1]
        elif opt == 'per_change':
            df[f'{f0}_{f1}_per_change'] = (df[f0]-df[f1]/(df[f1]+1e-7))*100
        elif opt == 'Z_score_res':
            df[f'{feat[0]}_{feat[3]}_zcore_res_gen'] = ((df[feat[0]]-df[feat[1]])/(df[feat[2]]+1e-7))

    #=================
    def _feature_eng_apply(self, df, config):
        cross_opts = {'ratio', 'diff_cross', 'multi', 'per_change', 'Z_score_res'}
        for opt, params in config.items():
            cols = params['cols']
            period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1
            for feature in cols:
                if opt in cross_opts:
                    self._gen_cross_feature(df, opt, feature, period)
                else:
                    self._gen_single_feature(df, opt, feature, period)
        return df 

    def transform(self, X):
        df = X.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = self._basic_feature(df)
        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values
        #df.index = pd.to_datetime(df.index)
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values
        df = self._fill_Nan(df, df.columns.tolist())

        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df