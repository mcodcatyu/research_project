import pandas as pd
import io
import os
import tempfile
#可以把這邊改成classs，然後一進來就直接根據他選的地方去做預測
#===========================
class GCMDprocessor:
    def __init__ (self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.df=None

    #def _get_file_content(self):
    #    if isinstance(self.uploaded_file, str):
    #        with open(self.uploaded_file, 'r', encoding='utf-8') as f:
    #            return f.read()
    #    else:
   #         content = self.uploaded_file.read().decode('utf-8')
    #        self.uploaded_file.seek(0)
#
    #    return content
    
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

    def parse_file(self):
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

    #def _predict(self, df):
        #預測的代碼放這

class OPTprocessor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
