import os
from sklearn.base import clone
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from data_convert import GCMDprocessor,TSFE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import glob
import joblib
import plotly.express as px
import traceback
from sqlalchemy import MetaData, Table, inspect, text
import xgboost as xgb
import lightgbm as lgb
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
            #else:
            #    processor = OPTICALprocessor(uploaded_file)
            #    df = processor._parse_file()

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
        def bagging_rf(n_bags, base_model, X_train_final, y_train):
            pos_idx = np.flatnonzero(y_train.to_numpy() == 1)
            unl_idx = np.flatnonzero(y_train.to_numpy() == 0)
            # obtain the number of positive and unlabeled data
            number_pos = len(pos_idx)

            rng = np.random.default_rng(42)
            models = []
            for bag in range(n_bags):
                sampled_pos = pos_idx

                #select same number data of positive data from unlabeld data
                sampled_unl_idx = rng.choice(
                    unl_idx,
                    size=number_pos,
                    replace=False
                )

                sampled_idx = np.concatenate([
                    pos_idx,
                    sampled_unl_idx
                ])


                rng.shuffle(sampled_idx)

                X_pu = X_train_final.iloc[sampled_idx]
                y_pu = y_train.iloc[sampled_idx]
                model_clone = clone(base_model)
                model_clone.fit(X_pu, y_pu)
                models.append(model_clone)
            return models
            
        def predict_proba_pu(models, X):
            probs = [m.predict_proba(X)[:, 1] for m in models]
            return np.mean(probs, axis=0)

        if 'evaluated' not in st.session_state:
            st.session_state.evaluated = False
        if 'eval_data' not in st.session_state:
            st.session_state.eval_data = {}

        if st.button('Train and Evaluate Model'):
            with st.spinner('Reading data from sql'):
                try:
                    df_train = pd.read_sql(f"SELECT * FROM `{train_table}`", con=engine)
                    if instrument_type == "GC-MD": # 
                        processor = GCMDprocessor(uploaded_file)
                        if df_train.empty:
                            st.warning("No data, please upload first!")
                        else:

                            #訓練的資料
                            st.success("data training...")
                            # 80% training, 20% testing
                            X_train_final, y_train_final, X_test_final, y_test_final = processor._preprocessing(df_train)
                            
                            #st.success(f'{X_train_final.columns}')
                            n_bags=30
                            base_rf = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=50, random_state = 42, n_jobs=1)
                            #base_rf =  xgb.XGBClassifier(
                             #learning_rate=0.1, max_depth=5, n_estimators= 200,
                            #random_state=42
                            #)
                            pu_models= bagging_rf(n_bags, base_rf , X_train_final, y_train_final)
                            model = pu_models
                            threshold = 0.8
                            name='XGBOOST'
                            y_prob_pu =  predict_proba_pu(pu_models, X_test_final)
                            y_pred_pu = (y_prob_pu >= threshold).astype(int)

                            auc_pu = roc_auc_score(y_test_final, y_prob_pu)
                            pr_auc_pu = average_precision_score(y_test_final, y_prob_pu)
                            
                            #====================_test_final)
                            fig_cm, axes_cm = plt.subplots(figsize=(8,6))
                            fig_roc, ax_roc = plt.subplots(figsize=(8,6))
                            fig_prc, ax_prc = plt.subplots(figsize=(8,6))
                            #====================
                            fpr_pu, tpr_pu, _ = roc_curve(y_test_final, y_prob_pu)
                            ax_roc.plot(fpr_pu, tpr_pu, linestyle='--', label=f'{name} (PU) - AUC: {auc_pu: .3f}')

                            prec_pu, rec_pu, _ = precision_recall_curve(y_test_final, y_prob_pu)
                            ax_prc.plot(rec_pu, prec_pu, linestyle='--', label=f'{name} (PU) - PU-AUC:{pr_auc_pu: .3f}')

                            cm = confusion_matrix(y_test_final, y_pred_pu)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm, cbar=False, xticklabels=['Pred:0','Pred:1'], yticklabels=['True:0', 'True:1'])
                            axes_cm.set_title(f"{name} (PU) Confusion matrix")
                            axes_cm.set_xlabel("Predicted Label")
                            axes_cm.set_ylabel("True Label")

                            ax_roc.plot([0, 1], [0,1], 'k--', alpha=0.5)
                            ax_roc.set_title("ROC CURVE: Standard vs PU learning")
                            ax_roc.set_xlabel("False Positive Rate")
                            ax_roc.set_ylabel("True Positive Rate")
                            ax_roc.legend()


                            fig_roc.tight_layout()
                            


                            baseline_pr = np.sum(y_test_final == 1) / len(y_test_final)

                            ax_prc.axhline(y=baseline_pr, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_pr:.3f})')
                            ax_prc.set_title("Precision-Recall Curve: Standard vs PU Learning")
                            ax_prc.set_xlabel("Recall")
                            ax_prc.set_ylabel("Precision")
                            ax_prc.legend()

                            st.session_state.eval_data={
                                'fig_cm':fig_cm,
                                'fig_roc':fig_roc,
                                'fig_prc':fig_prc,
                                'X_train_final':X_train_final,
                                'y_train_final': y_train_final,
                                'X_test_final': X_test_final,
                                'y_test_final': y_test_final,
                                'df_train_len':len(df_train),
                                'base_rf': base_rf,
                                'n_bags':n_bags
                            }
                            st.session_state.evaluated = True
                except Exception as ex:
                                    st.error(f"traiing failed:{ex}")
                                    st.code(traceback.format_exc())
        if st.session_state.evaluated:
            eval_data = st.session_state.eval_data
                            #=======================
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Confusion Matrix")
                st.pyplot(eval_data['fig_cm'], use_container_width=True)
            with col2:
                st.subheader("ROC Curve")
                st.pyplot(eval_data['fig_roc'], use_container_width=True)
            with col3:
                st.subheader("PR Curve")
                st.pyplot(eval_data['fig_prc'], use_container_width=True)
            st.markdown("---")
            if st.button('Training Final Model'):
                with st.spinner('Training final model on full dataset'):
                    X_train_all= pd.concat([eval_data['X_train_final'], eval_data['X_test_final']], axis=0).reset_index(drop=True)
                    y_train_all = pd.concat([eval_data['y_train_final'], eval_data['y_test_final']], axis=0).reset_index(drop=True)
                    final_models =  bagging_rf(eval_data['n_bags'], eval_data['base_rf'] , X_train_all, y_train_all)
 
                    timestamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
                    model_filename = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")
                    joblib.dump(final_models, model_filename)
                    st.success(f"model training complete! saved as {model_filename}")
                    st.info(f'this training used {eval_data["df_train_len"]}data record')
                        #====== feature importance
                    st.markdown("-----")
                    st.subheader("Feature Importance Analysis")
                    avg_imp = np.mean([m.feature_importances_ for m in final_models], axis=0)
                    avg_imp_pct = (avg_imp / np.sum(avg_imp)) * 100
                    df_imp = pd.DataFrame({
                        'Feature': X_train_all.columns,
                        'Importance': avg_imp_pct
                    })
                    df_imp_chart = df_imp.sort_values(by='Importance', ascending=True)

                    fig_imp = px.bar(
                        df_imp_chart,
                        x='Importance',
                        y='Feature',
                        title=f"Feature Imortantance(%)",
                        labels={'Importance':'Feature Importance (%)', 'Feature':'Feature Name'},
                        text_auto='.2f'
                    )
                    fig_imp.update_layout(height=500 + (30*30))

                    st.plotly_chart(fig_imp, use_container_width=True)
                    with st.expander("View all Feature Importance"):
                        df_imp_table = df_imp.sort_values(by='Importance',ascending=False).reset_index(drop=True)
                        st.dataframe(df_imp_table,  use_container_width=True)

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
                #elif instrument_type =="Optical":
                #    pred_processor = OPTICALprocessor(predict_file)
               
                df_test = pred_processor._parse_file() # df_test本身還含有std, air以外的data
                #st.success(f'{df_test.columns}')
                #看要不要出一個訊息說過濾那些數據?或是簡單點，寫 "只使用air和std的資料"
                valid_types = ['std', 'air']
                df_test_filtered = df_test[df_test['type'].isin(valid_types)].copy()

                X_test = pred_processor._predict(df_test_filtered)
               # st.success(f'{X_test.columns}')

                if st.button("Contact model prediction"):
                    all_probs = [m.predict_proba(X_test)[:, 1] for m in loaded_model]
                    probs = np.mean(all_probs, axis=0)


                    df_test_filtered['predicted_prob'] = pd.Series(probs, index=X_test.index)
                    df_results = df_test_filtered.sort_values(
                        by='predicted_prob', ascending=False
                    ).reset_index(drop=True)

                    st.session_state['pred_results'] = df_results

                    st.success("Prediction complete!")


                if 'pred_results' in st.session_state:
                    df_res = st.session_state['pred_results'].copy()

                    st.markdown("-----")

                    #st.header("Human - in- the loop")

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
                    df_sub = df_res.copy()
                    date_str = df_sub['date'].astype(str).str.zfill(6)
                    time_str = df_sub['time'].astype(str).str.zfill(6)

                    df_sub.index = pd.to_datetime(date_str + time_str, format='%y%m%d%H%M%S')

                    df_sub.index = pd.to_datetime(
                        df_sub.index.astype(str).str.replace('datetime', '').str.strip()
                    )
                    years = df_sub.index.year.unique()
                    df_sub['datetime'] = df_sub.index


                    def status(row, thresh):
                        if pd.isna(row['C']):
                            return 'Missing (NaN)'
                        elif row['predicted_prob'] >= thresh:
                            return f'Anomaly (>={threshold:.3f})'
                        else:
                            return 'Normal'


                    df_sub['status'] = df_sub.apply(status, axis=1, thresh=threshold)
                    df_sub['plot_val'] = df_sub['C'].fillna(1000)

                    fig = px.scatter(
                        df_sub,
                        x='datetime',
                        y='plot_val',
                        color = 'status',
                        symbol='status',
                        color_discrete_map = {
                            'Normal': '#41ad48',
                            'Missing (NaN)':'#b5b2b2',
                            f'Anomaly (>={threshold:.3f})':'#ffbd59'
                        },
                        symbol_map={
                            'Normal':'cross',
                            'Missing (NaN)':'diamond',
                            f'Anomaly (>={threshold:.3f})': 'circle'
                        },
                        hover_data = {
                            'datetime': '|%Y-%m-%d %H:%M:%S',
                            'C': ':.4f',
                            'predicted_prob': ':.4f',
                            'plot_val':False,
                            'status': True
                        },
                        labels={
                            'datetime':'Year',
                            'plot_val': 'Concentration',
                            'status': 'Category'
                        },
                        title = f'Concentration Distribution'

                    )
                    fig.update_xaxes(dtick="M12", tickformat="%Y", title_text="Year")

                    years = df_sub.index.year.unique()
                    for year in years:
                        start_year = df_sub[df_sub.index.year == year].index[0]
                        fig.add_vline(x=str(start_year), 
                                      line_width=1.5, 
                                      line_dash="dash", 
                                      line_color="black", 
                                      opacity=0.5,
                                      annotation_text= str(year),
                                      annotation_position='bottom left',
                                      annotation_font_size=12,
                                      annotation_font_color='black'
                                      )
                    st.plotly_chart(fig, use_container_width='True')
                    #fig.add_hline(
                    #    y = threshold,
                    #    line_dash = 'dash',
                    #    line_color = 'red',
                    #    annotation_text=f'Threshold = {threshold:.3f}',
                    #    annotation_position='top left'
                   #    
                    #)
                    #st.plotly_chart(fig, use_container_width=True)




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
                                        cols_to_drop = ['predicted_prob', 'datetime']
                            
                                        edited_df = edited_df.drop(columns=cols_to_drop, errors='ignore')

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