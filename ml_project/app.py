import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import numpy as np
from data_convert import GCMDprocessor
from sklearn.ensemble import RandomForestClassifier
import glob
import joblib

# 連sql
st.title('SQL Connection test')

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

#========
BASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:rootpassword@db:3306/",
)

# 連database
DB_URL = os.getenv(
    'DATABASE_URL',
    "mysql+pymysql://root:rootpassword@db:3306/mydatabase",
)
#===============================
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
#===========================
try:
    engine = get_engine(DB_URL) 
    with engine.connect() as conn:
        st.sidebar.success('MYSQL database')

    tab1, tab2, tab3, tab4 = st.tabs(
        ['1. upload dataset', 
         '2. preview database and tables',
         '3. Train Model',
         '4. Predict / Inference']) #兩個分頁
 
    #================
    with tab1:
        st.subheader('1. Produce data')

        uploaded_file = st.file_uploader("Select upload file", type=['txt'])
        if uploaded_file is not None:
            processor = GCMDprocessor(uploaded_file)
            df = processor.parse_file()# 檔案進來就解析

            st.write("Preview 100 :")

            #資料在這邊
            st.dataframe(df.head(100), use_container_width=True)
            target_table = st.text_input('Enter table name:', 'users')
            write_mode = st.radio("writing mode:", ["append","replace" ], format_func=lambda x: "Append" if x=='append' else "replace")


            if st.button('upload data into database'):   
                with st.spinner('Data uploading...' ):
                    df.to_sql(
                        name=target_table,
                        con=engine,
                        if_exists=write_mode,
                        index=True,
                        chunksize=10000,
                    )   

                    st.balloons() # 氣球圖案
                    st.success(f'uploaded {len(df):,} data to `{target_table}` table')
        else:
            st.info("Please upload txt file")

#=============== 
# Database Preview
#===================
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
                selected_table = st.selectbox('SELECT table:', table_list, key="table_select")
                
                st.markdown("---")
                st.write(
                    f"Current viewing {selected_db} -{selected_table}"
                )

                limit = st.slider(
                    "Preview limitation (Limit)",
                    min_value = 10,
                    max_value = 1000,
                    value=100,
                    step=10,
                )

                query = f"SELECT * FROM `{selected_table}` LIMIT {limit}"
                preview_df = pd.read_sql(query, con=current_db_engine)
                st.dataframe(preview_df, use_container_width=True)

    with tab3:
        st.subheader('Train model using DB Data')
        st.write("Training model by reading the cirrent exist latest data")
        with current_db_engine.connect() as conn:
            tables_df = pd.read_sql("SHOW TABLES", con=conn)
            table_list = tables_df.iloc[:, 0].tolist()
        train_table = st.selectbox('SELECT table:', table_list, key="train_table_input")

        if st.button('Restart training model'):
            with st.spinner('Reading data from sql'):
                try:
                    df_train = pd.read_sql(f"SELECT * FROM `{train_table}`", con=engine)

                    if df_train.empty:
                        st.warning("No data, please upload first!")
                    else:
                        #訓練的資料
                        X = df_train[:, :-1]
                        y = df_train[:, -1]

                        model = RandomForestClassifier()
                        model.fit(X,y)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_filename = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
                        joblib.dump(model, model_filename)

                        st.success(f"model training complete! saved as {model_filename}")
                        st.info(f'this training used {len(df_train):, }data record')
                except Exception as ex:
                    st.error(f"traiing failed")
    with tab4:
        st.subheader('Inference / Prediction')
        saved_models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl")), reverse=True)
        if not saved_models:
            st.warning("NO model can be used currently")
        else:
            selected_model_path = st.selectbox('select used model', options=saved_models)

            loaded_model =  joblib.load(selected_model_path)
            st.success("Success!")

            st.markdown("-------")

            predict_file = st.file_uploader("upload the target txt file", type=['txt'], key="pred_file")

            if predict_file is not None:
                pred_processor = GCMDprocessor(predict_file)
                df_test = pred_processor.parse_file()

                if st.button("Contact model prediction"):
                    predictions = loaded_model.predict(df_test)

                    df_test['Predicted_Result'] = predictions

                    st.subheader("Preview results")
                    st.dataframe(df_test.head(100), use_container_width=True)

                    csv_data = df_test.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='download all prediction results csv',
                        data = csv_data,
                        file_name = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime = "text/csv"
                    )
except Exception as e:
    st.error(f'connection failed: {e}')