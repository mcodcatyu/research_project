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
    def __init__ (self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.df=None

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
        target = 'flag_label' #這邊改成 label 1

        #df[] = df[].fillna(" ")
        #建立isc_nan label
        #df[target] = df['C'].isna().astype(int)
        #df['label_c_nan'] = df['C'].isna().astype(int)
        #只取用type = air 或是std的
        df = df[((df['type'] == 'air')|(df['type']=='std'))].copy()
       # print("df passed 1 step!")

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

            #=================================
        #feature_ml = ['level_area_ratio','ht_to_last_std_ratio',
        #'area_mad_3h','ht_roll_std_1h','w_roll_std_3h_w_roll_mean_3h_ratio',
        #'ht', 'skew','area_to_last_std_ratio',
        #'ht_roll_mean_1h','area_roll_std_1h',
        #'ht_pflow_ratio','ht_mad_3h','w_residual_3h',
        #'w_residual_1h', 'w_diff_1', 'rt_to_last_std_ratio','w_roll_std_1h', 'area_pflow_ratio',
        #'w_robust_residual_1h', 'ht_roll_median_1h', 'rt_to_last_air_ratio', 'last_std_ht',
        #'w_roll_mean_1h', 'w_residual_6h','w_roll_median_1h','area_roll_mean_1h',
        #'ht_roll_std_3h_ht_roll_mean_3h_ratio','w_roll_std_3h','rt_roll_mean_1h',]
        #=====================
        feature_cols= [
            'pflow', 'tmod',  'rt', 'w',
            'ht', 'area', 'skew', 'start_time', 'end_time',
            'start_level', 'end_level', 
            'last_std_ht',
            'last_air_ht', 'last_air_area', 'last_std_rt', 'last_air_rt', 'last_std_start_level',
            'last_air_start_level', 'ht_to_last_std_ratio', 'ht_to_last_air_ratio', 'area_to_last_std_ratio',
            'area_to_last_air_ratio', 'rt_to_last_std_ratio', 'rt_to_last_air_ratio', 'start_level_to_last_std_ratio',
            'start_level_to_last_air_ratio', 'duration_rt_ratio', 'rt_position',
            'baseline_slope', 'level_area_ratio'       ]

        feature_config ={
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



            #'log':{'cols':['last_std_CH4_ht','last_air_CH4_ht'
                        #'CH4_end_time', 
            #              ]},
            'roll_median':{'cols':['rt' ,
                                    'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h', '3D', '7D']},
            'roll_mad':{'cols':['rt' ,
                                    'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h', '3D', '7D']},
            'roll_median_percent_res':{'cols':['rt' ,
                                    'w', 'ht','area','pflow'],
                                    'period':['1h','3h','6h', '24h', '3D', '7D']},

        # 'multi':{'cols':[['CH4_w', 'CH4_ht'],['CH4_area', 'pflow']]},

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
        #預測的代碼放這
        X = df[feature_cols]
        y = df[target]
        split_idx = int(len(df)*0.8)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

        X_train_final = tsfe.fit_transform(X_train)
        #y_train_final = y_train.copy()

       # X_train_final  = X_train_final [feature_ml]
        X_test_final = tsfe.transform(X_test)
        #X_test_final  = X_test_final [feature_ml]
       # y_test_final = y_test.copy()
        #=================
    
        return X_train_final, y_train, X_test_final, y_test


    def _predict(self, df):
        #只取用type = air 或是std的

        df = df[((df['type'] == 'air')|(df['type']=='std'))].copy()
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

            #=================================
        #feature_ml = ['level_area_ratio','ht_to_last_std_ratio',
        #'area_mad_3h','ht_roll_std_1h','w_roll_std_3h_w_roll_mean_3h_ratio',
        #'ht', 'skew','area_to_last_std_ratio',
        #'ht_roll_mean_1h','area_roll_std_1h',
        #'ht_pflow_ratio','ht_mad_3h','w_residual_3h',
        #'w_residual_1h', 'w_diff_1', 'rt_to_last_std_ratio','w_roll_std_1h', 'area_pflow_ratio',
        #'w_robust_residual_1h', 'ht_roll_median_1h', 'rt_to_last_air_ratio', 'last_std_ht',
        #'w_roll_mean_1h', 'w_residual_6h','w_roll_median_1h','area_roll_mean_1h',
        #'ht_roll_std_3h_ht_roll_mean_3h_ratio','w_roll_std_3h','rt_roll_mean_1h',]
        #=====================
        feature_cols= [
            'pflow', 'tmod',  'rt', 'w',
            'ht', 'area', 'skew', 'start_time', 'end_time',
            'start_level', 'end_level', 
            'last_std_ht',
            'last_air_ht', 'last_air_area', 'last_std_rt', 'last_air_rt', 'last_std_start_level',
            'last_air_start_level', 'ht_to_last_std_ratio', 'ht_to_last_air_ratio', 'area_to_last_std_ratio',
            'area_to_last_air_ratio', 'rt_to_last_std_ratio', 'rt_to_last_air_ratio', 'start_level_to_last_std_ratio',
            'start_level_to_last_air_ratio', 'duration_rt_ratio', 'rt_position',
            'baseline_slope', 'level_area_ratio'       ]

        feature_config ={
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



            #'log':{'cols':['last_std_CH4_ht','last_air_CH4_ht'
                        #'CH4_end_time', 
            #              ]},
            'roll_median':{'cols':['rt' ,
                                    'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h', '3D', '7D']},
            'roll_mad':{'cols':['rt' ,
                                    'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h', '3D', '7D']},
            'roll_median_percent_res':{'cols':['rt' ,
                                    'w', 'ht','area','pflow'],
                                    'period':['1h','3h','6h', '24h', '3D', '7D']},

        # 'multi':{'cols':[['CH4_w', 'CH4_ht'],['CH4_area', 'pflow']]},

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
        #預測的代碼放這
        X_test = df[feature_cols]


        tsfe = TSFE(feature_cols=feature_cols, feature_config=feature_config)

        X_test_final = tsfe.fit_transform(X_test)
        #y_train_final = y_train.copy()

       # X_train_final  = X_train_final [feature_ml]
        #X_test_final  = X_test_final [feature_ml]
       # y_test_final = y_test.copy()
        #=================
    
        return X_test_final




class OPTICALprocessor:
    def __init__ (self, uploaded_file):
            self.uploaded_file = uploaded_file
            self.df=None

    @staticmethod
    def _split_flag (val):
        if pd.isna(val):
            return np.nan, np.nan
        
        val_str = str(val).strip()
        # check if last val is a character(Flag)
        if val_str[-1].isalpha() or val_str[-1] == '*':
            c_val = float(val_str[:-1])
            flag_val = val_str[-1]
        else:
            c_val = float(val_str) # normal value, then direcly convert to float
            flag_val = " " # no flag, give it a space
        
        return c_val, flag_val

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

            df_tac = pd.read_csv(file_path, skiprows=1, sep=r'\s+', header=1) 
            # skip first row, txt is space seperated

            df_tac['C'], df_tac['flag'] = zip(*df_tac['C'].map(self._split_flag))
            feature_name_tac = [
            'date', 'time', 'type', 'sample', 'standard', 'port', 
            'dry', 'wet', 'stdev', 'std_rep', 'std_stdev',
            'target_error','Cdrift', 'C', 'N', 'Nfiltered',
            'cycle_time', 'h2o', 'h2o_stdev', 'cavity_press', 'cavity_press_stdev', 
            'cavity_temp', 'cavity_temp_stdev', 'das_temp', 'etalon_temp', 
            'warmbox_temp', 'outlet_valve','flag'
            ]
            df_tac.columns = feature_name_tac

            self.df = self._process_datetime(df_tac)
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


class TSFE(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols,  feature_config):#, target_col='label1'):
        self.feature_cols = feature_cols
        self.feature_config = feature_config
    #==========
    def fit (self, X, y=None):
        return self
    #=============
    def _fill_Nan(self, df, feature_cols):
        df[feature_cols] = df[feature_cols].ffill().fillna(df[feature_cols].median()) # df[feature_cols].median()
        return df

    def _gen_single_feature(self, df, new_cols, opt, feature, period):
        if opt == 'diff':
            for p in period:
                new_cols[f'{feature}_diff_{p}'] = (new_cols[f'{feature}'].diff(p))
        elif opt == 'lag':
            for p in period:
                new_cols[f'{feature}_lag_{p}'] = (new_cols[f'{feature}'].shift(p))
        elif opt == 'roll_std':
            for p in period:
                new_cols[f'{feature}_roll_std_{p}'] = (new_cols[f'{feature}'].rolling(window=p, closed='left').std())
        elif opt == 'roll_mean_percent_res':
            for p in period:
                mean_col = df[f'{feature}'].rolling(window=p, closed='left').mean()
                new_cols[f'{feature}_roll_mean_{p}'] =mean_col #.fillna(df[f'{feature}_roll_mean_{p}'].median()) # self not included
                new_cols[f'{feature}_residual_{p}'] = ((new_cols[f'{feature}']- mean_col)/(mean_col+1e-7))*100 # self not included

        elif opt =='log':
            new_cols[f'{feature}_log'] = np.sign(new_cols[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))
        elif opt == 'relative_per':
            for p in period:
                roll = new_cols[f'{feature}'].rolling(window=p, closed='left')
                new_cols[f'{feature}_relative_per_{p}'] = (new_cols[f'{feature}']-(roll.min()))/((roll.max())-(roll.min())+1e-7)
        elif opt == 'per_rank':
            for p in period:
                new_cols[f'{feature}_per_rank_{p}'] = new_cols[f'{feature}'].rolling(window=p, closed='left').rank(pct=True)



    def _gen_cross_feature(self, df, new_cols, opt, feat, period):
        f0, f1 = feat[0], feat[1]
        def get_col(col_name):
            return(
                new_cols[col_name] if col_name in new_cols else df[col_name]
            )
        if opt == 'ratio':
            new_cols[f'{f0}_{f1}_ratio'] = get_col[f0]/get_col[f1]
        elif opt == 'diff_cross':
            new_cols[f'{f0}_{f1}_diff_cross'] = get_col[f0]-get_col[f1]
        elif opt == 'multi':
            new_cols[f'{f0}_{f1}_multi'] = get_col[f0]*get_col[f1]
        elif opt == 'per_change':
            new_cols[f'{f0}_{f1}_per_change'] = (get_col[f0]/get_col[f1]+1e-7)*100
        elif opt == 'Z_score_res':
            new_cols[f'{feat[0]}_{feat[3]}_zcore_res_gen'] = ((get_col[feat[0]]-get_col[feat[1]])/(get_col[feat[2]]+1e-7))

    #=================
    def _feature_eng_apply(self, df, config):
        cross_opts = {'ratio', 'diff_cross', 'multi', 'per_change', 'Z_score_res'}
        new_cols={}
        for opt, params in config.items():
            if opt not in cross_opts:
                cols = params['cols']
                period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1
                for feature in cols:
                    self._gen_single_feature(df, opt, new_cols, feature, period)

        for opt, params in config.items():
            if opt in cross_opts:
                cols = params['cols']
                period = params.get('period') or params.get('periods', 1) # period's value -> periods's -> 1
                for feature in cols:
                    self._gen_cross_feature(df, opt, new_cols, feature, period)

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)

        return df 

    def transform(self, X):
        df = X.copy(deep=False)
        df.index = pd.to_datetime(df.index)
        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values
        #df.index = pd.to_datetime(df.index)
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values
        df = self._fill_Nan(df, df.columns.tolist())

        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df