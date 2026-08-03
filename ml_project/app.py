import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from data_convert import GCMDprocessor,OPTICALprocessor,TSFE
from pulearn import ElkanotoPuClassifier
from sklearn.ensemble import RandomForestClassifier
import glob
import joblib
import plotly.express as px
import traceback
from sqlalchemy import MetaData, Table, inspect, text

# 連sql
st.title('GHG Data Anomaly detection')
#=========
MODEL_BASE_DIR = "models"
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
st.sidebar.header('Instrument setting')
instrument_type = st.sidebar.radio("Select instrument type:", ["GC-MD", "Optical"])
if instrument_type == "GC-MD":
    MODEL_DIR = os.path.join(MODEL_BASE_DIR,"instrument_GC-MD" )
    default_table_name = 'gcmd_dataset_1994_2025'

elif instrument_type=='Optical': 
    MODEL_DIR = os.path.join(MODEL_BASE_DIR,"instrument_optical")
    default_table_name = "optical_dataset_2013_2025"


os.makedirs(MODEL_DIR, exist_ok=True)

PROTECTED_MODELs = [""] # put the protect models here
#===============================
#快取，連線好不會因為刷新就一直重新連

@st.cache_resource
def get_engine(url):
    return create_engine(url, connect_args={"ssl_disabled":True})

def is_protected_model(filepath):
    filename = os.path.basename(filepath)
    return filename in PROTECTED_MODELs 
#===========================


try:
    engine = get_engine(DB_URL) 
    with engine.connect() as conn:
        st.sidebar.success('Database coonected')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ['1. upload dataset', 
         '2. preview database and tables',
         '3. Train Model',
         '4. Predict / Inference',
         '5. System Managemnet'
         ]) #兩個分頁
 
    #================
    with tab1:
        st.subheader('1. Produce data')

        uploaded_file = st.file_uploader("Select upload file", type=['txt'])
        if uploaded_file is not None:
            if instrument_type == "GC-MD":
                processor = GCMDprocessor(uploaded_file)
                df = processor._parse_file()# 檔案進來就解析
            else:
                processor = OPTICALprocessor(uploaded_file)
                df = processor._parse_file()

            # Preview
            st.write("Preview 100 :")

            #資料在這邊
            st.dataframe(df.head(100), width='stretch')
            target_table = st.text_input('Enter table name:', default_table_name)
            write_mode = st.radio("writing mode:", ["append","replace" ], format_func=lambda x: "Append" if x=='append' else "replace")


            if st.button('upload data into database'):   
                with st.spinner('Data uploading...' ):
                    df.to_sql(
                        name=target_table,
                        con=engine,
                        if_exists=write_mode,
                        index=False,
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
        st.subheader('Database Preview')

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

                number_query = f"SELECT COUNT(*) FROM `{selected_table}`"
                total_len = pd.read_sql(number_query, con=current_db_engine).iloc[0,0]
    
                st.write(
                    f"Total: {total_len}"
                )

                st.dataframe(preview_df, width='stretch')

    with tab3:
        st.subheader('Model training')
        st.write("Train the model using the data from the database")
        with engine.connect() as conn:
            tables_df = pd.read_sql("SHOW TABLES", con=conn)
            table_list = tables_df.iloc[:, 0].tolist()
        train_table = st.selectbox('SELECT table:', table_list, key="train_table_input")

        if st.button('Restart training model'):
            with st.spinner('Reading data from sql'):
                try:
                    df_train = pd.read_sql(f"SELECT * FROM `{train_table}`", con=engine)
                    if instrument_type == "GC-MD":
                        processor = GCMDprocessor(uploaded_file)
                        if df_train.empty:
                            st.warning("No data, please upload first!")
                        else:
                            #訓練的資料
                            st.success("data training...")
                            X_train_final, y_train_final, X_test_final, y_test_final = processor._preprocessing(df_train)
                            
                            #st.success(f'{X_train_final.columns}')
                            base_rf = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=12,
                                min_samples_leaf=5,
                                max_samples=0.8,
                                random_state=42
                            )
                    
                            pu_estimator = ElkanotoPuClassifier(
                                estimator=base_rf,
                                hold_out_ratio=0.2
                            )
                            model = pu_estimator
                            model.fit(X_train_final, y_train_final)
                            
                            timestamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
                            model_filename = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
                            joblib.dump(model, model_filename)
                            st.success(f"model training complete! saved as {model_filename}")
                            st.info(f'this training used {len(df_train) }data record')
                        #====== feature importance
                        st.markdown("-----")
                        st.subheader("Feature Importance Analysis")

                        importance_df = pd.DataFrame({
                            'Feature': X_train_final.columns,
                            'Importance': model.estimator.feature_importances_
                        }).sort_values(by='Importance', ascending=False)

                        top_n = 20
                        top_importance = importance_df.head(top_n).sort_values(by='Importance', ascending=True)

                        fig_imp = px.bar(
                            top_importance,
                            x='Importance',
                            y='Feature',
                            title=f"Top {top_n} Imortant Features",
                            labels={'Importance':'Feature Importance Score', 'Feature':'Feature Name'},
                            text_auto='.4f'
                        )
                        fig_imp.update_layout(height=500 + (top_n*20))

                        st.plotly_chart(fig_imp, width='stretch')
                        with st.expander("View all Feature Importance (top 20)"):
                            st.dataframe(importance_df.reset_index(drop=True), width='stretch')

                except Exception as ex:
                    st.error(f"traiing failed:{ex}")
                    st.code(traceback.format_exc())


#==================== Model Prediction
    with tab4:
        st.subheader('Inference / Prediction')
        saved_models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl")), reverse=True)
        if not saved_models:
            st.warning("No model can be used currently")
        else:
            selected_model_path = st.selectbox('select used model', options=saved_models)

            loaded_model =  joblib.load(selected_model_path)

            st.success("Success!")

            st.markdown("-------")

            predict_file = st.file_uploader("upload the target txt file",
                                             type=['txt'], key="pred_file")

            if predict_file is not None:
                if instrument_type == "GC-MD":
                    pred_processor = GCMDprocessor(predict_file)
                elif instrument_type =="Optical":
                    pred_processor = OPTICALprocessor(predict_file)
               
                df_test = pred_processor._parse_file() # df_test本身還含有std, air以外的data
                #st.success(f'{df_test.columns}')
                #看要不要出一個訊息說過濾那些數據?或是簡單點，寫 "只使用air和std的資料"
                valid_types = ['std', 'air']
                df_test_filtered = df_test[df_test['type'].isin(valid_types)].copy()

                X_test = pred_processor._predict(df_test)
               # st.success(f'{X_test.columns}')

                if st.button("Contact model prediction"):
                    probs = loaded_model.predict_proba(X_test)[:,1]

                    df_test_filtered['predicted_prob'] = probs

                    df_test_filtered = df_test_filtered.sort_values(
                        by="predicted_prob", ascending=False
                    ).reset_index(drop=False)
                    st.session_state['pred_results'] = df_test_filtered.sort_values(by='predicted_prob', ascending=False).reset_index(drop=True)

                    st.success("Prediction complete!")


                if 'pred_results' in st.session_state:
                    df_res = st.session_state['pred_results'].copy()

                    st.markdown("-----")

                    st.header("Human - in- the loop")

                    recommmended_thres = 0.5 #系統預設門檻
                    st.sidebar.info(f"Default Threshold:{recommmended_thres}")

                    #讓使用者可以在1~0範圍滑動threshold
                    threshold = st.slider(
                        "Adjust threshold",
                        min_value = 0.000,
                        max_value = 1.000,
                        value=recommmended_thres,
                        step=0.001
                    )

                    y_pred_binary = (df_res['predicted_prob'] >= threshold).astype(int)
                    selected_count = int(np.sum(y_pred_binary))
                    total_samples = len(df_res)
                    selected_ratio = (selected_count / total_samples)*100

                    col1, col2 = st.columns(2)
                    with col1:
                            st.metric(label='total_data', value=f"{total_samples:,}")
                    with col2:
                        st.metric(label='High prob/ positive', value=f"{selected_count:,}", delta=f"acount {selected_ratio:.2f}%")

                    st.markdown("--------------")

                    st.subheader("Data possibility distribution (Hover Original Index)")
                    #只抽樣5000，畫plotly
                    sample_size = min(5000, len(df_res))
                    df_sample = df_res.sample(n=sample_size, random_state=42).copy()
                    df_sample['data_index'] = df_sample.index

                    fig = px.scatter(
                        df_sample,
                        x='data_index',
                        y = 'predicted_prob',
                        hover_data = {
                            'data_index': True,
                            'predicted_prob':':.4f',
                        },

                        labels = {
                            'data_index': 'Data Index',
                            'predicted_prob': 'Predicted Probability',
                        },
                        title = f'Sample size {sample_size:,} Points Probability Distribution,probability > Threshold ({threshold:.3f}) potential anomaly data'
                    )

                    fig.add_hline(
                        y = threshold,
                        line_dash = 'dash',
                        line_color = 'red',
                        annotation_text=f'Threshold = {threshold:.3f}',
                        annotation_position='top left'
                        
                    )

                    st.plotly_chart(fig, width='stretch')
                    # human-in-the-loop

                    st.header("Human-in-te-loop")
                    st.write("View high probability data, select in the box of `human_label`:")

                    #df_res['human_label'] = (df_res['predicted_prob'] >= threshold).astype(int)

                    edited_df = st.data_editor(
                        df_res,
                        column_config={
                            "predicted_prob": st.column_config.NumberColumn("Model predicted probability", format="%.4f"),
                            #"human_label":st.column_config.SelectboxColumn(
                            #    "Human Label (0=normal; 1=anomaly)",
                            #    options=[1,0],
                            #    required=True
                            #)
                        },
                        width='stretch'
                    )
                    #================== data loading

                    st.markdown('Database setting')
                    inspector = inspect(engine)
                    existing_table = inspector.get_table_names()

                    save_mode = st.radio(
                        "Select save mode:",
                        options=["Append to exist tables", "Create new table"],
                        horizontal=True
                    ) 
                    target_save_table = ""

                    if save_mode == "Append to exist tables":
                        if existing_table:
                            target_save_table = st.selectbox("Please select append table", options=existing_table)
                        else:
                            st.warning("There is now tables in the database, pleases switch to 'Create New table'")
                    else:
                        col_db1, col_db2 = st.columns([2,1])
                        with col_db1:
                            input_table_name = st.text_input(
                                "Please enter the new table name:",
                                value = f"{default_table_name}_verified"
                            )
                            target_save_table=input_table_name.strip()
                        if target_save_table in existing_table:
                            st.error(f"Table `{target_save_table} exists! Please switch to append 'Current exist table'")


                    if st.button('Checked!Loading data into sql database'):
                        if not target_save_table:
                            st.error("Please fill in or select current table name!")
                        elif save_mode == "Create new table" and target_save_table in existing_table:
                            st.error(f"Failed! can not loading dat into `{target_save_table}` exists, change name")
                        else:
                            with st.spinner('Loading checked data into database'):
                                try:
                                    #time_col = ["date", "time"]

                                    if save_mode == "Append to exist tables":
                                        existing_df = pd.read_sql(f"SELECT * FROM `{target_save_table}`", con=engine)
                                        edited_df = edited_df.drop(columns=['predicted_prob', 'datetime'])
                                        combined_df = pd.concat(
                                        [existing_df, edited_df], ignore_index=False)

                                        if set(["date", "time"]).issubset(combined_df.columns):
                                            final_df = combined_df.drop_duplicates(subset=["date", "time"], keep='last')
                                        else:
                                            final_df = combined_df.drop_duplicates()

                                        final_df.to_sql(
                                            name=target_save_table,
                                            con=engine,
                                            if_exists="replace",
                                            index=False
                                        )
                                    else:
                                        final_df = edited_df
                                        final_df.to_sql(
                                            name=target_save_table,
                                            con=engine,
                                            if_exists='fail',
                                            index=False
                                        )
                                    st.balloons()
                                    st.success(f"Success!")
                                except Exception as e:
                                    st.error(f"Error:{e}")
                                    st.code(traceback.format_exc())

                    csv_data = df_test.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='download all prediction results csv',
                        data = csv_data,
                        file_name = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime = "text/csv"
                    )
#=========== tab5 - Management
    with tab5:
        st.subheader("System Management")
        m_col1, m_col2 = st.columns(2)

        #==== delete table
        with m_col1:
            st.markdown('Manage Database Tables')

            st.caption("Caution: Deleting a table is irreversible")

            try:
                inspector = inspect(engine)
                all_tables = inspector.get_table_names()

                if not all_tables:
                    st.info("No tables available in the current database.")
                else:
                    target_del_table = st.selectbox(
                        "Select table to DELETE",
                        options = all_tables,
                        key="del_table_select"
                    )

                    confirm_del_table = st.checkbox(
                        f' confirm delete table `{target_del_table}',
                        key = "chk_del_table"
                    )

                    if st.button("Delete Selected Table", type="primary", disabled=not confirm_del_table):
                        with st.spinner("Deleting table..."):
                            with engine.begin() as conn:
                                conn.execute(text(f"DROP TABLE `{target_del_table}`"))
                            st.success(f"Table `{target_del_table}` deleted successfully!")
                            st.rerun()
            except Exception as e:
                st.error(f'connection failed: {e}')
                st.code(traceback.format_exc())
        #===== dele model
        with m_col2:
            st.markdown("Manage Models")
            st.caption("Manage trained model files (.pkl)")

            all_models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl")), reverse=True)
            if not all_models:
                st.info("No saved Model")

            else:
                model_options = {os.path.basename(p): p for p in all_models}
                selected_model_name = st.selectbox(
                    "Select model to manage:",
                    options = list(model_options.keys()),
                    key="del_model_select"
                )

                selected_model_path = model_options[selected_model_name]

                if is_protected_model(selected_model_path):
                    st.warning("This is a System Default Model and cannot be deleted")
                else:
                    confirm_del_model = st.checkbox(
                        f"Confirm to delete model `{selected_model_name}`",
                        key="chk_del_model"
                    )

                    if st.button("Delete Selected Model", type="primary", disabled=not confirm_del_model):
                        try:
                            os.remove(selected_model_path)
                            st.success(f"Model `{selected_model_name}` deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f'connection failed: {e}')
                            st.code(traceback.format_exc())
    
except Exception as e:
    st.error(f'connection failed: {e}')
    st.code(traceback.format_exc())