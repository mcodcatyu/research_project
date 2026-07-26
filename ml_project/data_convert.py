import pandas as pd
#可以把這邊改成classs，然後一進來就直接根據他選的地方去做預測
#===========================

def parse_gcmd_file(file_path):
    parsed_rows = []

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

    feature_names = data_headers + flag_headers

    df_mhd=pd.read_fwf(
    file_path,
    skiprows=1,
    header=1,
    keep_default_na=False
    )



    df_mhd.columns = feature_names

    # read the file in the way that the flag space is right

    df_mhd_data = pd.read_csv(file_path, skiprows=1, sep=r'\s+',header=1, engine='python') # skip first row, txt is space seperated
    #這邊是為了 把flag和真實的data放一起，所以各只取flag 和flag以外的數值
    df_mhd_data.columns = feature_names

    real_data_mhd = df_mhd_data.iloc[:, :-4]

    df_mhd_flag = df_mhd.iloc[:, -4:]

    df_mhd_flag.index = real_data_mhd.index 
    real_data_mhd = pd.concat([real_data_mhd, df_mhd_flag], axis=1)
        
    return real_data_mhd

