from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import numpy as np

#1e-7 to prevent division by zero
#=====================================
# Previously tested features with low importance are commedted out (retained for future reference)

default_feature_config  ={
    'diff':{'cols': ['CH4_area', 
                    'CH4_ht','CH4_w' ,# 'CH4_rt' #'tmod',#'CH4_end_time','CH4_start_time',,'CH4_rt'
                ], 'periods':[1]},

    'lag':{'cols':[ 'CH4_ht', #'CH4_area', 'CH4_rt',  #'tmod',  'psamp','pamb','tamb',,'CH4_skew'
                #'pflow','CH4_w'
              ], 'periods':[1]},

    'diff_cross':{'cols':[#['CH4_ht', 'CH4_ht_lag_72'],
                          #['CH4_area', 'CH4_area_lag_72'],
                          ['CH4_end_time', 'CH4_start_time'],
                      # ['CH4_w_roll_mean_2h', 'CH4_w_roll_mean_7h'],
                        ['CH4_end_level', 'CH4_start_level'],
                        ['CH4_rt', 'pflow'],
                        #['CH4_ht', 'CH4_ht_lag_72'],['CH4_w', 'CH4_w_lag_72'],
                        #['pflow', 'pflow_lag_1'],['CH4_skew','CH4_skew_lag_1']
                        ]}, 

    'roll_std':{'cols': ['CH4_w', 'CH4_ht','CH4_area', 'CH4_end_time_CH4_start_time_diff_cross'], 'period':['1h','3h','6h', '24h']},

    'roll_mean_percent_res':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area',], 'period': ['1h','3h','6h','24h']},



    #'log':{'cols':['last_std_CH4_ht','last_air_CH4_ht'
                   #'CH4_end_time', 
     #              ]},
     'roll_median':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area', 'pflow'], 'period':['1h','3h','6h', '24h']},
     'roll_mad':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area', 'pflow'], 'period':['1h','3h','6h', '24h']},
     'roll_median_percent_res':{'cols':['CH4_rt' ,
                             'CH4_w', 'CH4_ht','CH4_area','pflow'],
                              'period':['1h','3h','6h', '24h']},

   # 'multi':{'cols':[['CH4_w', 'CH4_ht'],['CH4_area', 'pflow']]},

    'ratio':{'cols':[   ['CH4_skew','CH4_w'],
                        ['CH4_ht', 'pflow'],
                        ['CH4_ht', 'CH4_w'],
                        ['CH4_area', 'pflow'],
                        ['CH4_rt', 'pflow'],
                        ['CH4_ht', 'CH4_area'],
                        ['CH4_w', 'CH4_end_time_CH4_start_time_diff_cross'],
                      
                        ['CH4_start_level', 'CH4_end_level'],
                  
                        ['CH4_w_roll_std_3h', 'CH4_w_roll_mean_3h'], ['CH4_ht_roll_std_3h', 'CH4_ht_roll_mean_3h'],['CH4_area_roll_std_3h', 'CH4_area_roll_mean_3h'],
                        ['CH4_w_roll_std_24h', 'CH4_w_roll_mean_24h'], ['CH4_ht_roll_std_24h', 'CH4_ht_roll_mean_24h'],['CH4_area_roll_std_24h', 'CH4_area_roll_mean_24h'],
                       ]}, 


    'per_change':{'cols':[['CH4_ht_diff_1', 'CH4_ht_lag_1']]},
    #'relative_per': {'cols': ['CH4_w', 'CH4_ht'], 'period':['2h','7h']},
    #'Z_score_res':{'cols':[['CH4_w', 'CH4_w_roll_mean_3h', 'CH4_w_roll_std_3h', '3h'], #['CH4_end_time', 'CH4_end_time_roll_mean_7D', 'CH4_end_time_roll_std_7D', '7D']
                           #]},
    #'per_rank': {'cols': ['CH4_w', 'CH4_skew','CH4_ht'], 'period':['7D']},
    }

#==============================================================================
class TSFE:
    def __init__(self, feature_cols, feature_config=None, type_col='type'):
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


    #===================================================================================
    def _fill_Nan(self, df, feature_cols):
        """
        Forward-fill missing value(Nan), filling any remaining leading NaNs with the median.
        
        Args:
            df(pd.DataFrame): Input DataFrame containing raw features and numerical values
            feature_cols(list): List of column names to perform missing value imputation on.

        Returns:
            df(pd.DataFrame):DataFrame with all NaN values filled
        """
        fill_cols = [c for c in feature_cols if c!='type'] # Omit the 'type' column (string)
        df[fill_cols] = df[fill_cols].ffill().fillna(df[fill_cols].median()) 
        return df
    
    #===================================================================================
    def _basic_feature(self, df):
        """
        Generate features
        Args:
            df(pd.DataFrame):DataFrame containing raw features and numerical values

        Returns:
            df(pd.DataFrame):DataFrame with transformed and newly generated features
                - Added columns:'duration_rt_ratio', 'rt_position', 'baseline_slope', 'level_area_ratio','{col}_to_last_std_ratio',
        """
        target_cols = ['ht', 'area', 'rt', 'start_level']

        for col in target_cols:
            col = f'CH4_{col}'

            for t_type in ['std', 'air']: # Only calculate values for 'std' and 'air' types
                type_median = df.loc[df['type'] == t_type, col ].median() # Compute median for NaN imputation

                only_series = df[col].where(df['type']==t_type) 
                df[f'last_{t_type}_{col}'] = only_series.ffill().shift(1).fillna(type_median)

        # Features ratios relative to previous 'std' and 'air' values (e.g., current 'ht' to previous 'std' 'ht')
        for col in ['CH4_ht', 'CH4_area', 'CH4_rt', 'CH4_start_level']:
            df[f'{col}_to_last_std_ratio'] = df[col] / (df[f'last_std_{col}'] + 1e-7)
            df[f'{col}_to_last_air_ratio'] = df[col] / (df[f'last_air_{col}'] + 1e-7)

        # Measure relative peak broading
        df['duration_rt_ratio'] = (df['CH4_end_time'] -df['CH4_start_time'])/(df['CH4_rt']+ 1e-7)

        # Measure peak symmetry
        df['rt_position'] = (df['CH4_rt'] - df['CH4_start_time']) / (df['CH4_w']+ 1e-7)

        # Rate of baseline height change per unit time from peak start to end
        df['baseline_slope'] =(df['CH4_end_level'] -df['CH4_start_level'])/(df['CH4_w']+ 1e-7)
        
        # Ratio of background baseline height (taking the heigher value) to peak area
        df['level_area_ratio'] = np.maximum(df['CH4_end_level'],df['CH4_start_level'] )/(df['CH4_area']+1e-7)
        return df

    def _gen_single_feature(self, df, opt, feature, period=None):
        """
        Generate features based on a single column
        Args:
            df(pd.DataFrame):Target DataFrame to append new features to.
            opt (str) : Single-feature operation type (e.g., 'diff', 'lag', 'roll_std').            
            feature (list of str): List of column names participating in the single operation.
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
        # '{feature}_residual_{p}': Residual relative to rolling mean --> Percentage deviation from window mean
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
                    _clean(g.rolling(window=p, closed='left').median()))

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
            for p in period:
                median_col =df[f'{feature}_roll_median_{p}']
                df[f'{feature}_robust_residual_{p}'] = (
                    (df[f'{feature}']-median_col)/(median_col +1e-7 )*100
                )

    def _gen_cross_feature(self, df, opt, feat):
        """
            Generate features onvolving interactions between multiple columns
            Args:
                df(pd.DataFrame):Target DataFrame to append new features to
                opt (str) : Single-feature operation type (e.g., 'ratio', 'diff_cross', 'multi', 'per_change', 'Z_score_res')
                feat (list of str): List of column names participating in the cross operation.
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
            df[f'{f0}_{f1}_per_change'] = ((df[f0]-df[f1])/(df[f1]+1e-7))*100
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
                    self._gen_cross_feature(df, opt, feature)
                else:
                    self._gen_single_feature(df, opt, feature, period)
        return df 
    
    #================

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

        df = self._basic_feature(df)
        df = self._fill_Nan(df, self.feature_cols) # Fill Nan values

        df = self._feature_eng_apply(df, self.feature_config)

        df = df.replace([np.inf, -np.inf], np.nan) # handle inf values, prevent Nan values
        #Ensure all Nan values are filled
        df = self._fill_Nan(df, df.columns.tolist())

        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df
#===========================================
