import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# 連sql
st.title('SQL Connection test')
BASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:rootpassword@db:3306/",
)

#連database
DB_URL = os.getenv(
    'DATABASE_URL',
    "mysql+pymysql://root:rootpassword@db:3306/mydatabase",
)
#快取，連線好不會因為刷新就一直重新連

@st.cache_resource
def get_engine(url):
    return create_engine(url, connect_args={"ssl_disabled":True})

def load_data():
    #建立測試資料集，模擬的資料
    np.random.seed(42)
    n_neg, n_pos=139717, 15729 # 正樣本和負樣本數量
    y_neg = np.zeros(n_neg)
    y_pos = np.ones(n_pos)

    prob_neg = np.random.beta(1, 15, n_neg)
    prob_pos = np.random.beta(4, 4, n_pos)

    y_true = np.concatenate([y_neg, y_pos])
    y_prob = np.concatenate([prob_neg, prob_pos])
    return y_true, y_prob

def analysis_results():
    y_true, y_prob = load_data() #得到 y_true 和 y_prob(真實值和預測機率值)

    #紀錄原始 index
    df = pd.DataFrame({
        'original_index': np.arange(len(y_prob)),
        'true_label': y_true.astype(int),
        'probability':y_prob
    })
    return df

try:
    engine = get_engine(DB_URL) 
    with engine.connect() as conn:
        st.sidebar.success('MYSQL database')

    tab1, tab2 = st.tabs(['upload dataset', 'preview database and tables']) #兩個分頁
 
    #================
    with tab1:
        st.subheader('1. produce data')
        df = analysis_results() # 資料在這邊
        st.dataframe(df.head(100), use_container_width=True)
        target_table = st.text_input('Enter table name:', 'users')
  

        if st.button('upload data into database'):   
            with st.spinner('Data uploading...' ):
                df.to_sql(
                    name=target_table,
                    con=engine,
                    if_exists='append',
                    index=False,
                    chunksize=10000,
                )   

                st.balloons() # 氣球圖案
                st.success(f'uploaded {len(df):,} data to `{target_table} table')



    with tab2:
        st.subheader('MySQL database preview')

        base_engine = get_engine(BASE_URL)
        with base_engine.connect() as conn:
            databases_df = pd.read_sql("SHOW DATABASES;", con=conn)
            system_dbs = [
                "information_schema",
                "performance_schema",
                'mysql',
                "sys"
            ]

            db_list = [
                db
                for db in databases_df.iloc[:, 0].tolist()
                if db not in system_dbs
            ]

            selected_db = st.selectbox("SELECT database:", db_list)

            selected_db_url = f"mysql+pymysql://root:rootpassword@db:3306/{selected_db}"
            current_db_engine = get_engine(selected_db_url)

            with current_db_engine.connect() as conn:
                tables_df = pd.read_sql("SHOW TABLES", con=conn)
                table_list = tables_df.iloc[:, 0].tolist()

            if not table_list:
                st.info(f'Database `{selected_db} is empty')
            else:
                selecte_table = st.selectbox('SELECT table:', table_list, key="table_select")
                
                st.markdown("---")
                st.write(
                    f"Current viewing {selected_db} -{selecte_table}"
                )

                limit = st.slider(
                    "Preview limitation (Limit)",
                    min_value = 10,
                    max_value = 1000,
                    value=100,
                    step=10,
                )

                query = f"SELECT * FROM {selecte_table} LIMIT {limit}"
                preview_df = pd.read_sql(query, con=current_db_engine)
                st.dataframe(preview_df, use_container_width=True)
except Exception as e:
    st.error(f'connection failed: {e}')