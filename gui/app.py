import os
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)
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
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve

import glob
import joblib
import plotly.express as px
import traceback
from sqlalchemy import MetaData, Table, inspect, text
import xgboost as xgb
import lightgbm as lgb

#==============================Functions=====================================================
# tab1: Dataupload(only can upload data file the format is generated from GCWerks software)
# tab2: Preview database and tables (select the database and the table you'd like to check)
# tab3 : Model training/ Evaluation
# tab4: Anomaly probability prediction using saved models
# tab5: System management - Delete tables or model files
#====================================================================================

#******************************** Basic Settings ********************
# SQL connection
st.title('GHG Data Anomaly Detection')
#=========
MODEL_BASE_DIR = "models"
#========
BASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:rootpassword@db:3306/",
)

# Database Connection
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

# put the protect models here
PROTECTED_MODELs = ["default_model.joblib"]  


#****************************************************
@st.cache_resource
def get_engine(url):
    """
    Create and cache database connection engine
    Use Streamlit @st.cache_resource to prevent redudant connection overhead during page rerenders.

    Args:
        url (str) : Database connection string (URL)
    Returns:
        Engine: Connected database engine instance
    """

    return create_engine(url, connect_args={"ssl_disabled":True})

def is_protected_model(filepath):
    """
    Check if a model file is protected
    Args:
        filepath (str) : Model file name or file path to check
    Returns:
        Bool: True if the file is in the protected list, False otherwise 
    """
    filename = os.path.basename(filepath)
    return filename in PROTECTED_MODELs 

def bagging_rf(n_bags, base_model, X_train_final, y_train):
    """
        Conduct PU Bagging 
        Args:
            n_bags(int) : Number of bagging iterations (number of sub-models to train)
            base_model(estimator) : Base model(Here we use RandomForest)
            X_train_final(pd.DataFrame): Training data
            y_train (pd.Series): Training target labels

        Returns:
        models(list) : List containing n_bags fitted models
    """
    pos_idx = np.flatnonzero(y_train.to_numpy() == 1)
    unl_idx = np.flatnonzero(y_train.to_numpy() == 0)
    # Get sample counts for positive and unlabeled classes
    number_pos = len(pos_idx)

    rng = np.random.default_rng(42)
    models = []
    for bag in range(n_bags):

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
    """
    Calculate mean anomaly probability across PU Bagging sub-models for test set
        Args:
            models(List): List containing trained sub-models
            X(pd.DataFrame):Data for prediction
        Returns:
            np.ndarray: Mean anomaly probability for the positive class (labeled 1) across all sub-models
    """
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)

#=============================================
                    # GUI Section
#=============================================
try:
    # database connection 
    engine = get_engine(DB_URL) 
    with engine.connect() as conn:
        st.sidebar.success('Database coonected')


    # create 5 tab 
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ['1. Data Upload', 
         '2. Database & Tables Preview',
         '3. Model Training',
         '4. Model Prediction',
         '5. System Management'
         ]) 
 
#================= Tab 1 =============================
#               Upload data file
#=====================================================
    with tab1:
        st.subheader('1. Data Upload')

        uploaded_file = st.file_uploader("Select a file to upload", type=['txt']) # upload data file generated from GCWerks

        # File selected, start parse file
        if uploaded_file is not None:
            # Identify instrument type
            if instrument_type == "GC-MD":
                processor = GCMDprocessor(uploaded_file)
                df = processor._parse_file()# parse file
            #---------------- 
    
            # Preview data
            st.write("Preview 100: ")

            # Uploaded Data
            st.dataframe(df.head(100), width='stretch')
            target_table = st.text_input('Enter table name:', default_table_name)
            write_mode = st.radio("writing mode:", ["append","replace" ], format_func=lambda x: "Append" if x=='append' else "replace") # append or replace selection control part

            # upload data to database
            if st.button('Upload Data to Database'):   
                with st.spinner('Uploading Data...' ):
                    df.to_sql(
                        name=target_table,
                        con=engine,
                        if_exists=write_mode,
                        index=False,
                        chunksize=10000,
                    )   

                    st.balloons() # balloon partten
                    st.success(f'Successfully uploaded {len(df):,} rows to the `{target_table}` table')
        else:
            st.info("Please upload a TXT file")

#=============================== Tab 2===================================
                        # Database Preview
#========================================================================

    with tab2:
        st.subheader('Database & Tables Preview')

        base_engine = get_engine(BASE_URL) # get connection engine

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

            # show selection box
            selected_db = st.selectbox("Select Database:", db_list)

            selected_db_url = f"mysql+pymysql://root:rootpassword@db:3306/{selected_db}"
            current_db_engine = get_engine(selected_db_url)


            with current_db_engine.connect() as conn:
                tables_df = pd.read_sql("SHOW TABLES", con=conn)
                table_list = tables_df.iloc[:, 0].tolist()

            if not table_list:
                st.info(f'Database `{selected_db}` is empty')
            else:
                selected_table = st.selectbox('Select Table:', table_list, key="table_select")
                
                st.markdown("---")
                st.write(
                    f"Currently viewing {selected_db} -{selected_table}"
                )

                # *****data viewing number limitation *****
                limit = st.slider(
                    "Number of Rows to Display (Limit)",
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
                    f"Total Rows: {total_len}"
                )

                st.dataframe(preview_df, width='stretch')

#=============================Tab 3==========================================
#                         Model training                                            
#============================================================================
    with tab3:
        st.subheader('Model Training')
        st.write("Train the model using the data from the database")
        with engine.connect() as conn:
            tables_df = pd.read_sql("SHOW TABLES", con=conn)
            table_list = tables_df.iloc[:, 0].tolist()

        # select the training data from database tables    
        train_table = st.selectbox('Select Table:', table_list, key="train_table_input")

        # save state
        if 'evaluated' not in st.session_state:
            st.session_state.evaluated = False
        if 'eval_data' not in st.session_state:
            st.session_state.eval_data = {}

        #
        if st.button('Train and Evaluate Model'):
            with st.spinner('Fetching data from SQL...'):
                try:
                    df_train = pd.read_sql(f"SELECT * FROM `{train_table}`", con=engine)

                    if instrument_type == "GC-MD": # 
                        processor = GCMDprocessor(uploaded_file)
                        if df_train.empty:
                            st.warning("No data found. Please upload data first.")

                        #******************** Model training control part*******************
                        else:

                            # Training data
                            st.info("Training started")
                            # 80% training, 20% testing(evaluating)
                            X_train_final, y_train_final, X_test_final, y_test_final = processor._preprocess_train_data(df_train)
                            
                            #****** Model *****
                            n_bags=30
                            base_rf = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=50, random_state = 42, n_jobs=1)
                            #******************

                            pu_models= bagging_rf(n_bags, base_rf , X_train_final, y_train_final)
                            model = pu_models # final model

                            threshold = 0.8 # used to plot confusion metrix
                            name='LightGBM'
                            y_prob_pu =  predict_proba_pu(pu_models, X_test_final) # probability of as 1
                            y_pred_pu = (y_prob_pu >= threshold).astype(int) # label

                            auc_pu = roc_auc_score(y_test_final, y_prob_pu) # auc 
                            pr_auc_pu = average_precision_score(y_test_final, y_prob_pu) # prc
                            
                            #==================== Evaluation ==================
                            # create figures
                            fig_cm, axes_cm = plt.subplots(figsize=(8,6))
                            fig_roc, ax_roc = plt.subplots(figsize=(8,6))
                            fig_prc, ax_prc = plt.subplots(figsize=(8,6))
                            #====================
                            # Roc plot
                            fpr_pu, tpr_pu, _ = roc_curve(y_test_final, y_prob_pu)
                            ax_roc.plot(fpr_pu, tpr_pu, linestyle='--', label=f'{name} (PU) - AUC: {auc_pu: .3f}')

                            # Prc plot
                            prec_pu, rec_pu, _ = precision_recall_curve(y_test_final, y_prob_pu)
                            ax_prc.plot(rec_pu, prec_pu, linestyle='--', label=f'{name} (PU) - PU-AUC:{pr_auc_pu: .3f}')

                            # plot confusion metrix
                            cm = confusion_matrix(y_test_final, y_pred_pu)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm, cbar=False, xticklabels=['Pred:0','Pred:1'], yticklabels=['True:0', 'True:1'])

                            #========= auc, prc, confusion metrix figures setting======================
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
                            #===============================================================================
                            
                            # Save results
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
         #======================= Evaluation results ================
            col1, col2, col3 = st.columns(3)
            # ========= Col1: COnfusion metrix ===========
            with col1:
                st.subheader("Confusion Matrix")
                st.pyplot(eval_data['fig_cm'], use_container_width=True)
            #========== Col 2: ROC Curve ==================
            with col2:
                st.subheader("ROC Curve")
                st.pyplot(eval_data['fig_roc'], use_container_width=True)
            #============== Col3: PRC Curve ====================
            with col3:
                st.subheader("PRC Curve")
                st.pyplot(eval_data['fig_prc'], use_container_width=True)
         #=============================================================
            st.markdown("---")
            #================= Trainig on full dataset ===========================
            if st.button('Train Final Model'):
                with st.spinner('Train final model on full dataset...'):
                    # get all dataset
                    X_train_all= pd.concat([eval_data['X_train_final'], eval_data['X_test_final']], axis=0).reset_index(drop=True)
                    y_train_all = pd.concat([eval_data['y_train_final'], eval_data['y_test_final']], axis=0).reset_index(drop=True)

                    # train model
                    final_models =  bagging_rf(eval_data['n_bags'], eval_data['base_rf'] , X_train_all, y_train_all)
                    # timestamp: used as part of file name 
                    timestamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
                    model_filename = os.path.join(MODEL_DIR, f"model_{timestamp}.pkl")

                    # save model
                    joblib.dump(final_models, model_filename)

                    st.success(f"model training complete! saved as `{model_filename}`")
                    st.info(f'This training run used {eval_data["df_train_len"]} records')
                    #====================feature importance==========================
                    st.markdown("-----")
                    st.subheader("Feature Importance Analysis")
                    # average feature importance 
                    avg_imp = np.mean([m.feature_importances_ for m in final_models], axis=0)
                    # show as percentage
                    avg_imp_pct = (avg_imp / np.sum(avg_imp)) * 100
                    # save results
                    df_imp = pd.DataFrame({
                        'Feature': X_train_all.columns,
                        'Importance': avg_imp_pct
                    })
                    df_imp_chart = df_imp.sort_values(by='Importance', ascending=True)

                    # feature importance barchart
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

                    # show only numbers
                    with st.expander("View all Feature Importance"):
                        df_imp_table = df_imp.sort_values(by='Importance',ascending=False).reset_index(drop=True)
                        st.dataframe(df_imp_table,  use_container_width=True)

#=================================Tab 4 ========================================
#                            Model Prediction
#===============================================================================
    with tab4:
        st.subheader('Model Prediction')

        # Fetch existed models
        saved_models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl"))+glob.glob(os.path.join(MODEL_DIR, "*.joblib")), reverse=True)
        if not saved_models:
            st.warning("No trained model currently available. Please train a model first.")
        else:
            selected_model_path = st.selectbox('select used model', options=saved_models)

            # load selected model file
            loaded_model =  joblib.load(selected_model_path)

            st.success("Success!")

            st.markdown("-------")

            # upload prediction txt file
            predict_file = st.file_uploader("Upload target TXT file",
                                             type=['txt'], key="pred_file")

            if predict_file is not None:
                if instrument_type == "GC-MD":
                    pred_processor = GCMDprocessor(predict_file)
            
                df_test = pred_processor._parse_file() # parse file
              
                valid_types = ['std', 'air']
                df_test_filtered = df_test[df_test['type'].isin(valid_types)].copy()

                X_test = pred_processor._preprocess_test_data(df_test_filtered)
               

                if st.button("Run Model Prediction"):
                    all_probs = [m.predict_proba(X_test)[:, 1] for m in loaded_model] # 1 bag is a model, 30 bag is 30 model, so we need to get the probability of a data in 30 model
                    probs = np.mean(all_probs, axis=0) # average probability as an anomaly


                    df_test_filtered['predicted_prob'] = pd.Series(probs, index=X_test.index) # save predicted probability results
                    df_results = df_test_filtered.sort_values(
                        by='predicted_prob', ascending=False
                    ).reset_index(drop=True)

                    st.session_state['pred_results'] = df_results

                    st.success("Prediction complete!")


                if 'pred_results' in st.session_state:
                    df_res = st.session_state['pred_results'].copy()

                    st.markdown("-----")

                    #st.header("Human -in- the loop")

                    recommmended_thres = 0.5 # Default threshold
                    st.sidebar.info(f"Default Threshold: {recommmended_thres}")

                    #Allow users to adjust threshold via slider within 0 to 1 range
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

                    st.subheader("Data Probability Distribution (Hover Original Index)")
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

                    #=====================Concentration-Year fugure setting=================
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
                    # Identify year transitions and draw annual boundary lines on chart
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
                    #===========================================================

                    # human-in-the-loop

                    st.header("Human-in-the-loop")
                    st.write("Review high-probability predictions and manually update the `flag_label` column in the table:")

                    edited_df = st.data_editor(
                        df_res,
                        column_config={
                            "predicted_prob": st.column_config.NumberColumn("Model predicted probability", format="%.4f"),
                        },
                        width='stretch'
                    )

                    #================== data loading ===================
                    # Confirm data upload mode: append or replace

                    st.markdown('Database Settings')
                    inspector = inspect(engine)
                    existing_table = inspector.get_table_names()

                    save_mode = st.radio(
                        "Select save mode:",
                        options=["Append to existing tables", "Create new table"],
                        horizontal=True
                    ) 
                    target_save_table = ""

                    # verify table existence when user selects append mode
                    if save_mode == "Append to existing tables":
                        if existing_table:
                            target_save_table = st.selectbox("Please select append table", options=existing_table)
                        else:
                            st.warning("There is now tables in the database, pleases switch to 'Create New table'")

                    # Verify tanle existence when user selexts replace mode
                    else:
                        col_db1, col_db2 = st.columns([2,1])
                        with col_db1:
                            input_table_name = st.text_input(
                                "Please enter the new table name:",
                                value = f"{default_table_name}_verified"
                            )
                            target_save_table=input_table_name.strip()
                        if target_save_table in existing_table:
                            st.error(f"Table `{target_save_table} exists! Please switch to append 'Current existing table'")

                    # Execute data upload action based on user selection
                    if st.button('Save Checked Data to Database'):

                        if not target_save_table:
                            st.error("Please fill in or select current table name!")

                        elif save_mode == "Create new table" and target_save_table in existing_table:
                            st.error(f"Error: Table {target_save_table} already exists. Please choose a different name.")

                        else:
                            # Proceed with table upload once name validation succeeds
                            with st.spinner('Loading checked data into database'):
                                try:
                                    
                                    if save_mode == "Append to existing tables":
                                        existing_df = pd.read_sql(f"SELECT * FROM `{target_save_table}`", con=engine) # Fetch all record from existing table
                                        cols_to_drop = ['predicted_prob', 'datetime'] # Exclude test result columns from upload pipeline
                            
                                        edited_df = edited_df.drop(columns=cols_to_drop, errors='ignore')

                                        combined_df = pd.concat(
                                        [existing_df, edited_df], ignore_index=False) # Concatenate existing data with incoming upload dataset

                                        # Check for duplicate uploaded data and retain only the most recent entry
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
                                            if_exists='fail', # Raise error if table already exists
                                            index=False
                                        )
                                    st.balloons() 
                                    st.success(f"Success!")

                                # Display error message detailing cause on execution failure
                                except Exception as e:
                                    st.error(f"Error:{e}")
                                    st.code(traceback.format_exc())
                    #========= Enable CSV data download for users ==========
                    csv_data = df_test.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='Download All Prediction Results (CSV)',
                        data = csv_data,
                        file_name = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv", #Use timestamp as filename
                        mime = "text/csv"
                    )

#========================= tab5 ==================================
#                     Management
#=================================================================
    with tab5:
        st.subheader("System Management")

        # Use two separate tabs to manage tables and models
        m_col1, m_col2 = st.columns(2)

        #========== table deletion management
        with m_col1:
            st.markdown('Manage Database Tables')

            st.caption("Caution: Deleting a table is irreversible")

            try:
                inspector = inspect(engine)
                all_tables = inspector.get_table_names() # Fetch database tables

                if not all_tables:
                    st.info("No tables available in the current database.")
                else:
                    # Allow user to select table
                    target_del_table = st.selectbox(
                        "Select table to DELETE",
                        options = all_tables,
                        key="del_table_select"
                    )

                    # Confirmation checkbox setting
                    confirm_del_table = st.checkbox(
                        f' Confirm deletion of table `{target_del_table}`',
                        key = "chk_del_table"
                    )

                    # Execute deletion operation
                    if st.button("Delete Selected Table", type="primary", disabled=not confirm_del_table):
                        with st.spinner("Deleting table..."):
                            with engine.begin() as conn:
                                conn.execute(text(f"DROP TABLE `{target_del_table}` "))
                            st.success(f"Table `{target_del_table}` deleted successfully!")
                            st.rerun()

            # Display error message detailing cause on execution failure
            except Exception as e:
                st.error(f'connection failed: {e}')
                st.code(traceback.format_exc())

        #===== Model file management=======
        with m_col2:
            st.markdown("Manage Models")
            st.caption("View and manage saved model files (.pkl)")

            # Fetch all model file
            all_models = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl"))+glob.glob(os.path.join(MODEL_DIR, "*.joblib")), reverse=True)
            if not all_models:
                st.info("No saved models available")

            else:
                model_options = {os.path.basename(p): p for p in all_models}

                #Allow user to select model file
                selected_model_name = st.selectbox(
                    "Select model to manage:",
                    options = list(model_options.keys()),
                    key="del_model_select"
                )

                selected_model_path = model_options[selected_model_name]

                if is_protected_model(selected_model_path):
                    st.warning("This is a system default model and cannot be deleted")
                else:
                    # Deletion confirmation checkbox setting
                    confirm_del_model = st.checkbox(
                        f"Confirm deletion of model `{selected_model_name}`",
                        key="chk_del_model"
                    )

                    # Delet selected model file
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