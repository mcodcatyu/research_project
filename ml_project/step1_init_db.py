import os
import pandas as pd
from sqlalchemy import create_engine
from data_convert import parse_gcmd_file
#-------- 路徑與檔名
CSV_FILE_PATH = '../data/raw/mhd_ch4.txt' #檔名，後面換成使用者點選後的檔案路徑
TABLE_NAME = 'mhd_ch4_gcmd' # sql 裡面的table名字

# 資料處裡
df = parse_gcmd_file(CSV_FILE_PATH)
file_path = f'data/{TABLE_NAME}.csv'
df.to_csv(file_path, index=False)


#========

DB_URL = 'sqlite:///my_database.db' # 到時候把這邊的連結換掉
engine = create_engine(DB_URL)

print('STEP 1')

#-------
df = pd.read_csv(file_path)
print(
    f'Successed! data contains {len(df)}, columns{list(df.columns)}'
)

df.to_sql(name=TABLE_NAME, con=engine, if_exists='append', index=False)# 如果是使用這一上傳就
print(f'Data loaded into SQL data table [{TABLE_NAME}]')

#驗證資料庫裡面的資料筆數
#check_df = pd.read_sql(f'SELECT COUNT (*) as total FROM {TABLE_NAME}', engine)
#total_count = check_df['total'].iloc[0]
