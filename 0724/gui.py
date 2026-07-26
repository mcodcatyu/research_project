import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PASS = os.getenv('DB_PASS', 'rootpassward')

engine = create_engine(
    f'mysql+pymysql://root:{DB_PASS}@{DB_HOST}:3366/mydatabase'
)

MODEL_PATH = 'models/latest_model.pkl' # 模型路徑

st.title('ML data upload and auto-training')

tab1, tab2 = st.tabs(['1. upload to sql', '2. traigger model training'])

with tab1:
    st.header('upload new training data')
    uploaded_file = st.file_uploader('upload txt file', type=['txt'])

    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file) # 這邊讀取使用者上傳的資料
        st.write('file preview:', new_df.head())

        if st.button('Loding into SQL database'):
            try:
                new_df.to_sql(
                    'training_data',
                    con=engine,
                    if_exists='append',
                    index=False,
                )
                st.success('Insert new data into SQL database ')
                st.cache_data.clear()
            except Exception as e:
                st.error(f'Failed:{e}')
        st.divider()
        st.subheader('Current database data amount')

        try:
            total_df = pd.read_sql("SELECT * FROM training_data", con=engine)
            st.info(f"Current SQL database total contains **{len(total_df)}** training data")
        except Exception:
            st.warning('Current database has not created training_data data table')


with tab2:
    st.header('retraining model')

    if st.button('Extract SQL data and strat training'):
        with st.spinner('loadining SQL data...'):
            try:
                df = pd.read_sql("SELECT * FROM training _data", con = engine)

                #這邊放模型訓練 
               #-----
                model = RandomForestClassifier()
               # 訓練模型儲存
                os.makedirs('models', exist_ok=True)
                joblib.dump(model, MODEL_PATH)

                st.success(f'model training completed')
                st.info(f'Newest model has been uploaded to : {MODEL_PATH}')
            except Exception as e:
                st.error(f'training precoss error occured: {e}')