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


#===========================
class GCMDprocessor:
    def __init__ (self, uploaded_file, type_col='type'):
        """
            GCMD 資料處裡入口
            Args:

            Returns:

        """
        self.uploaded_file = uploaded_file
        self.df=None
        self.type_col = type_col
        self.feature_cols= [
                    'pflow', 'tmod',  'rt', 'w','type',
                    'ht', 'area', 'skew', 'start_time', 'end_time',
                    'start_level', 'end_level', 
                ]
        
        #********* 使用的先前挑選出的重要性前30個的特徵 ****************
        self.feature_ml = ['ht_roll_std_24h_ht_roll_mean_24h_ratio',
                            'rt_to_last_air_ratio',
                            'area_roll_std_24h_area_roll_mean_24h_ratio',
                            'duration_rt_ratio',
                            'rt_position',
                            'ht_to_last_std_ratio',
                            'area_roll_std_3h_area_roll_mean_3h_ratio',
                            'w_roll_std_24h_w_roll_mean_24h_ratio',
                            'ht_roll_std_3h_ht_roll_mean_3h_ratio',
                            'end_time',
                            'w_residual_6h',
                            'rt_pflow_ratio',
                            'level_area_ratio',
                            'w_robust_residual_3h',
                            'w_residual_3h',
                            'w_robust_residual_6h',
                            'area_robust_residual_1h',
                            'area_residual_1h',
                            'area_to_last_std_ratio',
                            'w_roll_std_3h_w_roll_mean_3h_ratio',
                            'start_level_to_last_std_ratio',
                            'area_diff_1',
                            'ht_pflow_ratio',
                            'ht_diff_1_ht_lag_1_per_change',
                            'w_residual_24h',
                            'ht_residual_1h',
                            'w_diff_1',
                            'w',
                            'start_time',
                            'area_to_last_air_ratio']
        
        #********* 特徵工程設定的config ****************
        self.feature_config ={
                    'diff':{'cols': ['area', 
                                    'ht','w' , #'rt' 
                                ], 'periods':[1]},
        
                    'lag':{'cols':['ht', #'area', 'rt',   'pflow','w'
                                    ], 
                                'periods':[1]},
        
                    #'diff_cross':{'cols':[
                    #                    ['end_time', 'start_time'],
                    #                    ['end_level', 'start_level'],
                    #                    ['rt', 'pflow'],
                    #                   ]}, 
        
                    'roll_std':{'cols': ['w', 'ht','area', #'end_time_start_time_diff_cross'
                                         ], 
                                'period':['1h','3h','6h', '24h']},
        
                    'roll_mean_percent_res':{'cols':['rt' ,
                                            'w', 'ht','area',], 'period': ['1h','3h','6h','24h']},
        
        
        
                    #'log':{'cols':['last_std_ht','last_air_ht'
                                #'end_time', 
                    #              ]},
                    'roll_median':{'cols':['rt' ,
                                            'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h']},
                   #'roll_mad':{'cols':['rt' ,
                    #                        'w', 'ht','area', 'pflow'], 'period':['1h','3h','6h', '24h']},
                    'roll_median_percent_res':{'cols':['rt' ,
                                            'w', 'ht','area','pflow'],
                                            'period':['1h','3h','6h', '24h']},
        
                # 'multi':{'cols':[['w', 'ht'],['area', 'pflow']]},
        
                    'ratio':{'cols':[   ['skew','w'],
                                        ['ht', 'pflow'],
                                        ['ht', 'w'],
                                        ['area', 'pflow'],
                                        ['rt', 'pflow'],
                                        ['ht', 'area'],
                                        #['w', 'end_time_start_time_diff_cross'],
        
                                        ['start_level', 'end_level'],
                                
                                        ['w_roll_std_3h', 'w_roll_mean_3h'], ['ht_roll_std_3h', 'ht_roll_mean_3h'],['area_roll_std_3h', 'area_roll_mean_3h'],
                                        ['w_roll_std_24h', 'w_roll_mean_24h'], ['ht_roll_std_24h', 'ht_roll_mean_24h'],['area_roll_std_24h', 'area_roll_mean_24h'],
                                    ]}, 
        
        
                    'per_change':{'cols':[['ht_diff_1', 'ht_lag_1']]},
                    #'relative_per': {'cols': ['w', 'ht'], 'period':['2h','7h']},
                    #'Z_score_res':{'cols':[['w', 'w_roll_mean_3h', 'w_roll_std_3h', '3h'], 
                    #                    ]},
                    #'per_rank': {'cols': ['w', 'skew','ht'], 'period':['7D']
                    #             },
                    }

        
    # name the first 'ht' column into "inlet column"
    def _fix_feature_names(self, names):
        """
            Args:

            Returns:
        """
        unique_names=[]
        ht_seen=False

        for name in names:
            if name == 'ht' and not ht_seen:
                unique_names.append('inlet_ht')
                ht_seen=True
            else:
                unique_names.append(name)

        return unique_names
    
    #================================================
    # data parse here
    # The GC-MD file format is quite complex, that contains space and fixed-width
    # so we use below method to ensure the data is correct readed
    #================================================
    def _parse_file(self):
        """
            Args:

            Returns:

        """
        # Determine the source of input file: physical path strings or memory file objects
        

        if isinstance(self.uploaded_file, str):
            file_path = self.uploaded_file
            cleanup_temp = False # original file, no need to delete after execution

        # memory file: create temorary physical file with the same extension name for subsequent tools to read
        else:
            suffix = os.path.splitext(self.uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp_file:
                tmp_file.write(self.uploaded_file.getvalue())
                file_path = tmp_file.name
            cleanup_temp = True

        #================ Logic ============
        # obtain full header line -> obtain data header part & flag header(Spaces specify splitting required)
        # -> obtain final feature name -> read data, read flag data -> combine feature and data
        #===============
        try:
            flag_total_len = 28
            with open(file_path, 'r', encoding='utf-8') as f:
            #skip header
                lines = f.readlines()
            header_line = lines[2].rstrip('\r\n')

            # Split data header and flag header
            header_data_part = header_line[:-flag_total_len ]
            header_flag_part = header_line[-flag_total_len:]

            # header preprocess
            data_headers = header_data_part.strip().split()
            flag_headers = [
                header_flag_part[0:8].strip(),
                header_flag_part[8:16].strip(),
                header_flag_part[16:21].strip(),
                header_flag_part[21:28].strip()
            ] 

            # obtain final feature names
            raw_feature_names = data_headers + flag_headers

            # name first 'ht' as 'inlet_ht'(where it should be)
            feature_names = self._fix_feature_names(raw_feature_names)

            # read data(we will only use the data part without flag data part after)
            df_mhd=pd.read_fwf(
            file_path,
            skiprows=3,
            header=None,
            names=feature_names,
            keep_default_na=False
            )

            # set feature name as the previous obtained feature name(we will only use flag data for this one after)
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

            # obtain instrument data from df_mhd_data
            real_data_mhd = df_mhd_data.iloc[:, :-4]

            # obtain flag data from df_mhd_data
            df_mhd_flag = df_mhd.iloc[:, -4:]

            df_mhd_flag.index = real_data_mhd.index 

            # 將儀器資料與flag資料連接成一個df
            real_data_mhd = pd.concat([real_data_mhd, df_mhd_flag], axis=1)
            # fill with " "(讀取時空格會被當作nan 因此在這邊補上空格)
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

            # 建立機器學習需要的 label 欄位
            real_data_mhd ["flag_label"] = ((real_data_mhd ["flag_ht_encod"]==1) | (real_data_mhd ["flag_a_encod"]==1)).astype(int)
            real_data_mhd=real_data_mhd.drop(columns=["flag_a_encod", "flag_ht_encod"])
            self.df = self._process_datetime(real_data_mhd)

            return self.df
        finally:
            # 若為寫入之臨時檔，則程式結束後自動刪除
            if cleanup_temp and os.path.exists(file_path):
                os.remove(file_path)


    # 設定 timestamp作為 index(後續 Feature_eng 需要)
    def _process_datetime(self, df):
        """
            Args:

            Returns:

        """
        df['date'] = df['date'].astype(str).str.zfill(6)
        df['time'] = df['time'].astype(str).str.zfill(6)
                
                
        df['datetime_str'] = df['date'] + df['time']

        df['datetime'] = pd.to_datetime(df['datetime_str'], format='%y%m%d%H%M%S')

        df.set_index('datetime', inplace=True)
        df.drop(columns=['datetime_str'], inplace=True, errors='ignore')
        return df 

    # 
    def _preprocess_train_data(self, df):
        """
            Args:

            Returns:

        """
        # 目標 label
        target = 'flag_label'        
        #預測的代碼放這
        X = df[self.feature_cols]
        y = df[target]
        split_idx = int(len(df)*0.8)

        # 依照時間順序分為 80% train; 20% test.
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        tsfe = TSFE(feature_cols=self.feature_cols, feature_config=self.feature_config)

        # 特徵工程轉換
        X_train_final = tsfe.transform(X_train)
        X_test_final = tsfe.transform(X_test)

        # 只用以設定好的特徵
        X_train_final = X_train_final[self.feature_ml]
        X_test_final = X_test_final[self.feature_ml]
        #=================
        return X_train_final, y_train, X_test_final, y_test

    def _preprocess_test_data(self, df):
        """
            test data 前處理與特徵轉換
            Args:

            Returns:

        """
        X_pred = df[self.feature_cols]
        tsfe = TSFE(feature_cols=self.feature_cols, feature_config=self.feature_config)

        X_pred_final = tsfe.transform(X_pred)
        X_pred_final = X_pred_final[self.feature_ml]
        return X_pred_final

#============================== TSFE(Time-Series Feature Engineering) ========================================================
class TSFE:
    def __init__(self, feature_cols,  feature_config, type_col='type'):
        """
             class TSFE 入口
             Args:
                 feature_cols (list):選擇使用的原始特徵欄位
                 feature_config (dict): 轉換的特徵類別以及要產生的特徵
                 type_col (string, optional): 樣本類型
        """
        self.feature_cols = feature_cols
        self.feature_config = feature_config
        self.type_col = type_col

    #==========
    def _fill_Nan(self, df, feature_cols):
        """
        fill missing value(Nan), 前向填充後以中位數填充(若前向為nan時)
        Args:
            df(pd.DataFrame): 包含原始特徵與數值的輸入資料
            feature_cols (list): 轉換的特徵類別以及要產生的特徵

        Returns:
            df(pd.DataFrame):完成NAN值填充後的資料。
        """
        fill_cols = [c for c in feature_cols if c!='type']
        df[fill_cols] = df[fill_cols].ffill().fillna(df[fill_cols].median()) 
        return df
    #===================
    def _generate_base_features(self, df):
        """
        Args:
            df(pd.DataFrame):包含原始特徵與數值的輸入資料

        Returns:
            df(pd.DataFrame):完成特徵轉換與新增後的資料。新增欄位:
        """
        #========
        target_cols = ['ht', 'area', 'rt', 'start_level']

        for col in target_cols:
            for t_type in ['std', 'air']:# 只計算std和 air類型的數值
                type_median = df.loc[df['type'] == t_type, col].median()# 計算中位數，用於填補 NAN值

                only_series = df[col].where(df['type']==t_type)
                df[f'last_{t_type}_{col}'] = only_series.ffill().shift(1).fillna(type_median)

        # 特徵和前一個std以及air的同特徵的數值比。譬如: 當前ht和 前一個std type的數值比較。
        for col in ['ht', 'area', 'rt', 'start_level']:
            df[f'{col}_to_last_std_ratio'] = df[col] / (df[f'last_std_{col}'] + 1e-7)
            df[f'{col}_to_last_air_ratio'] = df[col] / (df[f'last_air_{col}'] + 1e-7)
        # 衡量峰的相對展寬程度
        df['duration_rt_ratio'] = (df['end_time'] -df['start_time'])/df['rt']

        # 衡量峰的對稱性
        df['rt_position'] = (df['rt'] - df['start_time']) / df['w']

        # 峰開始到結束的基線高度單位時間變化
        df['baseline_slope'] =(df['end_level'] -df['start_level'])/df['w']

        # 背景基線高度(取較高者)相對於峰面積的比例
        df['level_area_ratio'] = np.maximum(df['end_level'],df['start_level'] )/(df['area']+1e-7)

        return df
    
    #===============
    def _gen_single_feature(self, df, opt, feature, period=None):
        """
            Args:
                df(pd.DataFrame):包含原始特徵與數值的輸入資料

            Returns:
                df(pd.DataFrame):完成特徵轉換與新增後的資料。新增欄位:
        """

        #同一個type自己跟自己比
        if self.type_col in df.columns:
            g = df.groupby(self.type_col)[feature]
        else:
            g = df[feature]

        def _clean(reset):
            """
                Args:
                    df(pd.DataFrame):包含原始特徵與數值的輸入資料

                Returns:
                    df(pd.DataFrame):完成特徵轉換與新增後的資料。新增欄位:
            """
             # 清理由groupby計算後產生的多重索引，並還原成原始DataFrame的順序
            if isinstance(reset.index, pd.MultiIndex):
                return reset.reset_index(0, drop=True).sort_index()
            return reset

        # 特徵自己和自己前p個值的差異
        if opt == 'diff':
            for p in period:
                df[f'{feature}_diff_{p}'] = _clean(g.diff(p))

        # 特徵自己的前p個值
        elif opt == 'lag':
            for p in period:
                df[f'{feature}_lag_{p}'] = _clean(g.shift(p))

        # 特徵自己 p時間範圍內的 standard deviation
        elif opt == 'roll_std':
            for p in period:
                df[f'{feature}_roll_std_{p}'] = _clean(g.rolling(window=p, closed='left').std())

        # '{feature}_roll_mean_{p}':特徵自己在p時間範圍內的平均值
        # '{feature}_residual_{p}': 以及特徵當下值和此平均值的差值
        elif opt == 'roll_mean_percent_res':
            for p in period:
                mean_col = _clean(g.rolling(window=p, closed='left').mean())
                df[f'{feature}_roll_mean_{p}'] =mean_col #.fillna(df[f'{feature}_roll_mean_{p}'].median()) # self not included
                df[f'{feature}_residual_{p}'] = ((df[f'{feature}']- mean_col)/(mean_col+1e-7))*100 # self not included

        # 特徵自己的log 值
        elif opt =='log':
            df[f'{feature}_log'] = np.sign(df[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))

        # 當前數值在過去p區間(最高值到最低值)的相對位置比例
        elif opt == 'relative_per':
            for p in period:
                roll_min = _clean(g.rolling(window=p, closed='left').min())
                roll_max = _clean(g.rolling(window=p, closed='left').max())
                df[f'{feature}_relative_per_{p}'] = (df[f'{feature}']-(roll_min))/((roll_max)-(roll_min)+1e-7)

        # 在p時間範圍內的排名
        elif opt == 'per_rank':
            for p in period:
                df[f'{feature}_per_rank_{p}'] = _clean(g.rolling(window=p, closed='left').rank(pct=True))

        # 在p時間範圍內的 中位數值
        elif opt == 'roll_median':
            for p in period:
                df[f'{feature}_roll_median_{p}'] = (
                    _clean(g.rolling(window=p, closed='left').median())
                )

        # 過去p時間範圍內的中位數絕對偏差(Median Absolute Deviation)
        elif opt == 'roll_mad':
            def _calc_mad(x):
                med = np.median(x)
                return np.median(np.abs(x-med))

            for p in period:
                df[f'{feature}_mad_{p}']=(
                    _clean(g.rolling(window=p, closed='left').apply(_calc_mad, raw=True))
                )

        # 健壯殘差比(Robust Residual Percentage)，當前數值相對於過去p時間範圍中位數偏離了百分之多少
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
        """
            Args:
                df(pd.DataFrame):包含原始特徵與數值的輸入資料
            Returns:
                df(pd.DataFrame):完成特徵轉換與新增後的資料。新增欄位:
        """
        f0, f1 = feat[0], feat[1]

        # 特徵f0 和 f1的比值
        if opt == 'ratio':
            df[f'{f0}_{f1}_ratio'] = df[f0]/df[f1]

        # 特徵 f0 和 特徵 f1 的差植
        elif opt == 'diff_cross':
            df[f'{f0}_{f1}_diff_cross'] = df[f0]-df[f1]

        # 特徵 f0 和 特徵 f1 數值相乘
        elif opt == 'multi':
            df[f'{f0}_{f1}_multi'] = df[f0]*df[f1]

        #特徵 f0 相較於特徵f1的百分比變化
        elif opt == 'per_change':
            df[f'{f0}_{f1}_per_change'] = (df[f0]-df[f1]/(df[f1]+1e-7))*100

        # Z分數殘差
        elif opt == 'Z_score_res':
            df[f'{feat[0]}_{feat[3]}_zcore_res_gen'] = ((df[feat[0]]-df[feat[1]])/(df[feat[2]]+1e-7))

    #=================
    def _feature_eng_apply(self, df, config):
        """
            Args:
                df(pd.DataFrame):包含原始特徵與數值的輸入資料
                config(dict): 包含所有特徵內容的字典

            Returns:
                df(pd.DataFrame):完成特徵轉換與新增後的資料。新增欄位:
        """
        
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
        """
            Args:
                X(pd.DataFrame):包含原始特徵與數值的輸入資料

            Returns:
                df(pd.DataFrame):完成特徵轉換與新增後的資料。新增欄位:
        """
        df = X.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = self._generate_base_features(df)
        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values

        
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values

        #確保NAN數值皆填充
        df = self._fill_Nan(df, df.columns.tolist())

        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df