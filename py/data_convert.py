import pandas as pd
#可以把這邊改成classs，然後一進來就直接根據他選的地方去做預測
#===========================

def parse_mhd_file(file_path, feature_names):
    parsed_rows = []

    flag_total_len = 29

    with open(file_path, 'r', encoding='utf-8') as f:
        #skip header
        lines = f.readlines()

    header_line = lines[0].rstrip('\r\n')

    header_data_part = header_line[:-flag_total_len ]
    header_flag_part = header_line[-flag_total_len:]

    data_headers = header_data_part.strip().split()
    flag_headers = [
        header_flag_part[0:8].strip(),
        header_flag_part[8:15].strip(),
        header_flag_part[15:22].strip(),
        header_flag_part[22:29].strip()
    ] 

    feature_names = data_headers + flag_headers
    data_lines = lines[2:]
    for line in data_lines:
        line = line.rstrip('\r\n')

        if not line.strip():
            continue
        # split data & flag  part

        data_part = line[:-flag_total_len]
        flag_part = line[-flag_total_len:]

        data_values = data_part.strip().split()

        #flag area
        f_ht = flag_part[0:8].strip() 
        f_a = flag_part[8:15].strip()
        f_ =flag_part[15:22].strip()
        f_p = flag_part[22:29].strip()


        flags = [f_ht,f_a,f_ ,f_p]

        parsed_rows.append(data_values + flags)

    df = pd.DataFrame(parsed_rows, columns=feature_names)
    return df