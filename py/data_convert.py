import pandas as pd
#可以把這邊改成classs，然後一進來就直接根據他選的地方去做預測

file_path = "../data/raw/mhd_ch4.txt"

df_mhd=pd.read_fwf(
    file_path,
    skiprows=1,
    header=1,
    keep_default_na=False
)
df_mhd
import pandas as pd
pd.set_option('display.max_columns', None)

feature_names = [  
'date', 'time', 'type', 'sample', 'standard', 'port', 'ht', 'tmod', 'tamb', 'lab_temp', 'pflow', 'psamp', 'ploop', 'pamb', 'CH4_rt', 'CH4_w', 
'CH4_ht', 'CH4_area', 'CH4_skew', 'CH4_start_time', 'CH4_end_time', 'CH4_start_level', 'CH4_end_level', 'CH4_Rl_ht', 'CH4_Rl_a', 'CH4_Rl', 'CH4_norm_ht', 'CH4_norm_a', 'CH4_norm_w', 
'CH4_R_ht', 'CH4_R_a', 'CH4_R', 'CH4_C_ht', 'CH4_C_a', 'CH4_C', 'CH4_std_stdev', 'CH4_std_rep', 'CH4_Cstd', 'CH4_flag_ht', 'CH4_flag_a', 'CH4_flag', 'CH4_flag_p'
]
df_mhd.columns = feature_names

# read the file in the way that the flag space is right
import pandas as pd

file_path = "../data/raw/mhd_ch4.txt"

df_mhd_csvred = pd.read_csv(file_path, skiprows=1, sep=r'\s+',header=1, engine='python') # skip first row, txt is space seperated

feature_names = [  
'date', 'time', 'type', 'sample', 'standard', 'port', 'ht', 'tmod', 'tamb', 'lab_temp', 'pflow', 'psamp', 'ploop', 'pamb', 'CH4_rt', 'CH4_w', 
'CH4_ht', 'CH4_area', 'CH4_skew', 'CH4_start_time', 'CH4_end_time', 'CH4_start_level', 'CH4_end_level', 'CH4_Rl_ht', 'CH4_Rl_a', 'CH4_Rl', 'CH4_norm_ht', 'CH4_norm_a', 'CH4_norm_w', 
'CH4_R_ht', 'CH4_R_a', 'CH4_R', 'CH4_C_ht', 'CH4_C_a', 'CH4_C', 'CH4_std_stdev', 'CH4_std_rep', 'CH4_Cstd', 'CH4_flag_ht', 'CH4_flag_a', 'CH4_flag', 'CH4_flag_p'
]
df_mhd_csvred.columns = feature_names

df_mhd_csvred['type'].unique()
real_data_mhd = df_mhd_csvred.iloc[:, :-4]

real_data_mhd.head(20)

df_mhd_flag = df_mhd.iloc[:, -4:]
df_mhd_flag.head(20)

df_mhd_flag.index = real_data_mhd.index 
real_data_mhd = pd.concat([real_data_mhd, df_mhd_flag], axis=1)
real_data_mhd.head(20)

real_data_mhd.to_csv("../data/processed/mhd_ch4_formatted_v1.csv", index=False)

#===== optical
import pandas as pd

file_path = "../data/raw/tac_ch4.txt"

df_tac = pd.read_csv(file_path, skiprows=1, sep=r'\s+', header=1) 
# skip first row, txt is space seperated

import numpy as np

def split_flag (val):
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


df_tac['C'], df_tac['ch4_flag'] = zip(*df_tac['C'].map(split_flag))

print(df_tac[['C', 'ch4_flag']].head(20))
df_tac['C']

feature_name_tacch4 = [
  'date', 'time', 'type', 'sample', 'standard', 'port', 
 'ch4_dry', 'ch4_wet', 'ch4_stdev', 'ch4_std_rep', 'ch4_std_stdev',
  'ch4_target_error','ch4_Cdrift', 'ch4_C', 'ch4_N', 'ch4_Nfiltered',
  'cycle_time', 'h2o', 'h2o_stdev', 'cavity_press', 'cavity_press_stdev', 
  'cavity_temp', 'cavity_temp_stdev', 'das_temp', 'etalon_temp', 
  'warmbox_temp', 'outlet_valve','ch4_flag'
]


df_tac.columns = feature_name_tacch4

df_tac.to_csv("../data/processed/tac_ch4_formatted_v1.csv", index=False)