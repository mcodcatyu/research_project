import pandas as pd
import os
import tempfile
import numpy as np
import pandas as pd
#1e-7 to prevent division by zero
#=====================================
# Previously tested features with low importance are commedted out (retained for future reference)

default_feature_cols=[
                    'pflow', 'tmod',  'rt', 'w','type',
                    'ht', 'area', 'skew', 'start_time', 'end_time',
                    'start_level', 'end_level', 
                ]

default_feature_ml = ['ht_roll_std_24h_ht_roll_mean_24h_ratio',
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


default_feature_config = {
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

        

#===========================
class GCMDprocessor:
    def __init__ (self, uploaded_file, type_col='type', feature_cols = None, feature_ml = None, feature_config=None):
        """
            Process GC-MD instrment raw data (prduced by GCwerks)
            Args:
                uploaded_file:Path or file-like object of the uploaded raw data
                type_col(str, optional): Column name Representing the sample type. Defaults to 'type'
                feature_cols(list):List of raw feature column names to use.
                feature_ml(list, optional): List of selected features. Default to top 30 important feature
                feature_config(dict, optional): Configuration mapping feature categories to specific features to generate. Uses default settings if  None.
        """
        self.uploaded_file = uploaded_file
        self.type_col = type_col
        self.df=None #Strore parsed file data for downstream processsing
        self.feature_cols= feature_cols if feature_cols is not None else default_feature_cols
        
        #**Allow custom feature list input; fallback to top 30 pre-selected important features if not provided **
        self.feature_ml = feature_ml if feature_ml is not None else default_feature_ml
        
        self.feature_config = feature_config if feature_config is not None else default_feature_config
    # name the first 'ht' column into "inlet column"
    def _fix_feature_names(self, names):
        """
            Specifically renames the first occurrence of 'ht' to 'inlet_ht'.
            
            Args:
                names(list): original feature names

            Returns: 
                unique_names(list): List of modified feature names.
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
            Parse uploaded GC-MD TXT file.
            Separates raw measurements data from flags, and performs label encoding and time feature processing
            Args:
                None: Uses instance attribute self.uploaded_file directly
            Returns:
                pd.DataFrame: Processed instrument DataFrame with generated 'flag_label' column.
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
        #===================================
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
            #Extract flag and non-flag data separatel to combine flags with actual measurements 
            
            df_mhd_data.columns = feature_names

            # obtain instrument data from df_mhd_data
            real_data_mhd = df_mhd_data.iloc[:, :-4]

            # obtain flag data from df_mhd_data
            df_mhd_flag = df_mhd.iloc[:, -4:]

            df_mhd_flag.index = real_data_mhd.index 

            # Concatenate instrument data and flag data into a single DataFrame
            real_data_mhd = pd.concat([real_data_mhd, df_mhd_flag], axis=1)
            # fill with " "(treated as NaN during parsing)
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

            # Create target label column required for ML 
            real_data_mhd ["flag_label"] = ((real_data_mhd ["flag_ht_encod"]==1) | (real_data_mhd ["flag_a_encod"]==1)).astype(int)
            real_data_mhd=real_data_mhd.drop(columns=["flag_a_encod", "flag_ht_encod"])
            self.df = self._process_datetime(real_data_mhd)

            return self.df
        finally:
            # Auto-delete temporary file upon programme exit
            if cleanup_temp and os.path.exists(file_path):
                os.remove(file_path)


    # Set timestamp as index (required for feature engineering)
    def _process_datetime(self, df):
        """
            Combine date and time columns into a DatetimeIndex
            Merges 'date'(YYMMDD) and 'time' (HHMMSS columns and converts them into a DatetimeIndex
            Args:
                df(pd.DataFrame): Input DataFrame containing 'date' and 'time' columns

            Returns:
                df(pd.DataFrame): DataFrame indexed by datetime (DatetimeIndex)

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
        Split data chronologically and apply TSFE ransformation.
        Splits input data into an 80% training set and a 20% validation set by time, then applies time-series feature engineering.    
            Args:
                df(pd.DataFrame): Complete DataFrame containing raw features and target labels.
            Returns:
                tuple: (X_train_final, y_train, X_test_final, y_test)
                X_train_final: Transformed training features
                y_train: Training target labels
                X_test_final:Transformed validation features
                y_test:Validation target labels

        """
        # Target label
        target = 'flag_label'        
        
        X = df[self.feature_cols]
        y = df[target]
        split_idx = int(len(df)*0.8)

        # Chronological split: 80% train, 20% test
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        tsfe = TSFE(feature_cols=self.feature_cols, feature_config=self.feature_config)

        # Feature engineering transformation
        X_train_final = tsfe.transform(X_train)
        X_test_final = tsfe.transform(X_test)

        # Select only pre-configured features
        X_train_final = X_train_final[self.feature_ml]
        X_test_final = X_test_final[self.feature_ml]
        #=================
        return X_train_final, y_train, X_test_final, y_test

    def _preprocess_test_data(self, df):
        """
        Apply TSFE transformation and feature selection to prediction data

            Args:
                df(pd.Dataframe): Raw input data for prediction

            Returns:
                pd.DataFrame: Transformed and filtered feature DataFrame (X_pred_final) ready for inference

        """
        X_pred = df[self.feature_cols]
        tsfe = TSFE(feature_cols=self.feature_cols, feature_config=self.feature_config)

        X_pred_final = tsfe.transform(X_pred)
        X_pred_final = X_pred_final[self.feature_ml]
        return X_pred_final

#============================== TSFE(Time-Series Feature Engineering) ========================================================
class TSFE:
    def __init__(self, feature_cols,  feature_config=None, type_col='type'):
        """
            Initialize the TSFE class.
            Args:
                feature_cols(list):List of raw feature column names to use.
                feature_config(dict, optional): Configuration mapping feature categories to specific features to generate. Uses default settings if  None.
                type_col(str, optional): Column name Representing the sample type. Defaults to 'type'
        """
        self.feature_cols = feature_cols

        if feature_config is not None:
            self.feature_config = feature_config
        else:
            self.feature_config = default_feature_config 
       
        self.type_col = type_col

    #==========
    def _fill_Nan(self, df, feature_cols):
        """
        Forward-fill missing value(Nan),  filling any remaining leading NaNs with the median.
        
        Args:
            df(pd.DataFrame): Input DataFrame containing raw features and numerical values
            feature_cols(list): List of column names to perform missing value imputation on.

        Returns:
            df(pd.DataFrame):DataFrame with all NaN values filled
        """
        fill_cols = [c for c in feature_cols if c!='type']
        df[fill_cols] = df[fill_cols].ffill().fillna(df[fill_cols].median()) 
        return df
    #===================
    def _generate_base_features(self, df):
        """
        Generate features
        Args:
             df(pd.DataFrame):DataFrame containing raw features and numerical values
 
        Returns:
             df(pd.DataFrame):DataFrame with transformed and newly generated features
                 - Added columns:'duration_rt_ratio', 'rt_position', 'baseline_slope', 'level_area_ratio','{col}_to_last_std_ratio',
        """
        #========
        target_cols = ['ht', 'area', 'rt', 'start_level']

        for col in target_cols:
            for t_type in ['std', 'air']:# Only calculate values for 'std' and 'air' types
                type_median = df.loc[df['type'] == t_type, col].median()# Compute median for NaN imputation

                only_series = df[col].where(df['type']==t_type)
                df[f'last_{t_type}_{col}'] = only_series.ffill().shift(1).fillna(type_median)

        # Features ratios relative to previous 'std' and 'air' values (e.g., current 'ht' to previous 'std' 'ht')
        for col in ['ht', 'area', 'rt', 'start_level']:
            df[f'{col}_to_last_std_ratio'] = df[col] / (df[f'last_std_{col}'] + 1e-7)
            df[f'{col}_to_last_air_ratio'] = df[col] / (df[f'last_air_{col}'] + 1e-7)
        # Measure relative peak broading
        df['duration_rt_ratio'] = (df['end_time'] -df['start_time'])/df['rt']

        # Measure peak symmetry
        df['rt_position'] = (df['rt'] - df['start_time']) / df['w']

        #Rate of baseline height change per unit time from peak start to end
        df['baseline_slope'] =(df['end_level'] -df['start_level'])/df['w']

        # Ratio of background baseline height (taking the heigher value) to peak area
        df['level_area_ratio'] = np.maximum(df['end_level'],df['start_level'] )/(df['area']+1e-7)

        return df
    
    #===============
    def _gen_single_feature(self, df, opt, feature, period=None):
        """
        Generate features based on a single column
        Args:
            df(pd.DataFrame):Target DataFrame to append new features to.
            opt (str) : Single-feature operation type (e.g., 'diff', 'lag', 'roll_std').            feature:
            period (int, str, or list): Time window or  period step(s) for calculation.
        Returns:
            df(pd.DataFrame):Dataframe updated with newly engineered features.
            Added columns: 'diff', 'lag', 'roll_std', 'roll_mean_percent_res', 'log', 'relative_per','per_rank', 'roll_median', 'roll_mad',
            'roll_median_percent_res', 
        """

        # perform group-wise calculations within the same sample type
        if self.type_col in df.columns:
            g = df.groupby(self.type_col)[feature]
        else:
            g = df[feature]

        def _clean(reset):
            """
                            Clean MultiIndex resulting from group-by operations and restore original index order
                            Args:
                                reset(pd.DataFrame): Intermediate DataFrame resulting from grouped calculations
            
                            Returns:
                                reset(pd.DataFrame): Dataframe sorted back to its original index order
                        """
                        # Clean up MultiIndex created by groupby restore original DataFrame order
            if isinstance(reset.index, pd.MultiIndex):
                return reset.reset_index(0, drop=True).sort_index()
            return reset

        # Difference between current feature value and its value p periods ago
        if opt == 'diff':
            for p in period:
                df[f'{feature}_diff_{p}'] = _clean(g.diff(p))

        # Lagged feature value by period p
        elif opt == 'lag':
            for p in period:
                df[f'{feature}_lag_{p}'] = _clean(g.shift(p))

        # Rolling standard deviation over time window p
        elif opt == 'roll_std':
            for p in period:
                df[f'{feature}_roll_std_{p}'] = _clean(g.rolling(window=p, closed='left').std())

        # '{feature}_roll_mean_{p}':Rolling mean over time window p
        # '{feature}_residual_{p}': Residual relative to rolling mean --> Percentage deviation from win
        elif opt == 'roll_mean_percent_res':
            for p in period:
                mean_col = _clean(g.rolling(window=p, closed='left').mean())
                df[f'{feature}_roll_mean_{p}'] =mean_col 
                df[f'{feature}_residual_{p}'] = ((df[f'{feature}']- mean_col)/(mean_col+1e-7))*100 

        # Log transformation of feature
        elif opt =='log':
            df[f'{feature}_log'] = np.sign(df[f'{feature}'])*np.log1p(np.abs(df[f'{feature}']))

        # Relative position (min-max ratio) of current value within past window p
        elif opt == 'relative_per':
            for p in period:
                roll_min = _clean(g.rolling(window=p, closed='left').min())
                roll_max = _clean(g.rolling(window=p, closed='left').max())
                df[f'{feature}_relative_per_{p}'] = (df[f'{feature}']-(roll_min))/((roll_max)-(roll_min)+1e-7)

        # Rolling percentile rank over window p
        elif opt == 'per_rank':
            for p in period:
                df[f'{feature}_per_rank_{p}'] = _clean(g.rolling(window=p, closed='left').rank(pct=True))

        # Rolling median over time window p
        elif opt == 'roll_median':
            for p in period:
                df[f'{feature}_roll_median_{p}'] = (
                    _clean(g.rolling(window=p, closed='left').median())
                )

        # Median Absolute Deviation over time window p
        elif opt == 'roll_mad':
            def _calc_mad(x):
                med = np.median(x)
                return np.median(np.abs(x-med))

            for p in period:
                df[f'{feature}_mad_{p}']=(
                    _clean(g.rolling(window=p, closed='left').apply(_calc_mad, raw=True))
                )

        # Robust Residual Percentage: Percentage deviation of current value relative to rolling median over window p 
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
            Generate features onvolving interactions between multiple columns
            Args:
                df(pd.DataFrame):Target DataFrame to append new features to
                opt (str) : Single-feature operation type (e.g., 'ratio', 'diff_cross', 'multi', 'per_change', 'Z_score_res')
                feature (list of str): List of column names participating in the cross operation.
                period (int, str, or list): Time window or period step(s) for calculation.
            returns:
                df(pd.DataFrame):Dataframe updated with newly engineered features.
                Added columns:'ratio', 'diff_cross', 'multi', 'per_change', 'Z_score_res'
        """
        f0, f1 = feat[0], feat[1]

        # Ratio feature f0 to feature f1
        if opt == 'ratio':
            df[f'{f0}_{f1}_ratio'] = df[f0]/df[f1]

        # Difference between feature f0 and feature f1
        elif opt == 'diff_cross':
            df[f'{f0}_{f1}_diff_cross'] = df[f0]-df[f1]

        # Product of feature f0 and feaure f1
        elif opt == 'multi':
            df[f'{f0}_{f1}_multi'] = df[f0]*df[f1]

        # Percentage change of feature f0 relative to feature f1
        elif opt == 'per_change':
            df[f'{f0}_{f1}_per_change'] = (df[f0]-df[f1]/(df[f1]+1e-7))*100

        # Z-score residual
        elif opt == 'Z_score_res':
            df[f'{feat[0]}_{feat[3]}_zcore_res_gen'] = ((df[feat[0]]-df[feat[1]])/(df[feat[2]]+1e-7))

    #=================
    def _feature_eng_apply(self, df, config):
        """
            Apply feature engineering rules defined in the config dictionary
            Args:
                df(pd.DataFrame):Input DataFrame containing raw features and numerical values
                config(dict): Feature engineering configuration mapping operations (str) to
                                their  corresponding parameter dictionaries(containing 'cols', 'period', or 'periods').

            Returns:
                df(pd.DataFrame):Dataframe updated with newly engineered features.
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
            Transform raw input data through the complete feature eng pipeline
            Args:
                X(pd.DataFrame):Raw input DataFrame containing initial time-series features.

            Returns:
                df(pd.DataFrame): Tranformed DataFrame with engineered features, cleaned missing and potential inf values, and optimized float32 datatypes
        """
        df = X.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = self._generate_base_features(df)
        df = self._fill_Nan(df, self.feature_cols) # fill for original feature values

        
        df = self._feature_eng_apply(df, self.feature_config)
        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values

        #Ensure all Nan values are filled
        df = self._fill_Nan(df, df.columns.tolist())

        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df