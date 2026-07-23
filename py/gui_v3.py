import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

st.set_page_config(layout='wide')
st.title("threshold and numbers")
st.write('try difffernt threshold')

#==========
def load_data():
    #建立測試資料集
    np.random.seed(42)
    n_neg, n_pos=139717, 15729 # 正樣本和負樣本數量
    y_neg = np.zeros(n_neg)
    y_pos = np.ones(n_pos)

    prob_neg = np.random.beta(1, 15, n_neg)
    prob_pos = np.random.beta(4, 4, n_pos)

    y_true = np.concatenate([y_neg, y_pos])
    y_prob = np.concatenate([prob_neg, prob_pos])
    return y_true, y_prob

#====================
def analysis_results():
    y_true, y_prob = load_data() #得到 y_true 和 y_prob(真實值和預測機率值)

    #紀錄原始 index
    df = pd.DataFrame({
        'original_index': np.arange(len(y_prob)),
        'true_label': y_true.astype(int),
        'probability':y_prob
    })


    #計算權範圍門檻的精準度、召回率
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)



    #========== Sidebar
    st.sidebar.header("Control screeen")

    recommmended_thres = 0.4694 #系統預設門檻
    st.sidebar.info(f"Recommend best threshold:{recommmended_thres}")

    #讓使用者可以在1~0範圍滑動threshold
    threshold = st.sidebar.slider(
        "Adjust threshold",
        min_value = 0.000,
        max_value = 1.000,
        value=recommmended_thres,
        step=0.001
    )


    y_proba = (y_prob >=threshold).astype(int) #預測標籤(預測值)

    total_samples = len(y_true)
    selected_count = int(np.sum(y_proba)) #判斷為正樣本的總數量
    selected_ratio = (selected_count / total_samples) * 100#判斷為正樣本的總數量占總資料的百分比 

    idx = np.argmin(np.abs(thresholds- threshold))
    current_prec = precisions[idx]
    current_rec= recalls[idx]
    current_f1 = 2 * (current_prec * current_rec) / (current_prec + current_rec + 1e-7)

    #動態指標與顯示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label='selected positive labels', value=f"{selected_count}", delta=f"%in total {selected_ratio:.4f}%")
    with col2:
        st.metric(label='Current Precision', value=f"{current_prec:.4f}")
    with col3:
        st.metric(label='Recall / F1-score', value=f"{current_rec:.4f}", delta=f"F1:{current_f1:.4f}")

    st.markdown("----------")

    col_left, col_right = st.columns(2)

    with col_left: # 下方畫面左側，柱狀圖顯示當前的Precision, Recall 與 F1-score
        st.subheader("Bar Chart")

        metrics_df = pd.DataFrame({
            'metric': ['Precision', 'Recall', 'F1-score'],
            'value': [current_prec, current_rec, current_f1]
        })
        st.bar_chart(data=metrics_df, x='metric', y='value')

    with col_right:#下方畫面右側，繪製PR取線，用紅點標註出當前門檻值在曲線上的對應位置。
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(recalls, precisions, label='Model PR Curve')

        ax.scatter(current_rec, current_prec, color='red', s=100, zorder=5, label=f'Current (T={threshold:.2g})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)


    #=========== Plotly 機率分布圖
    st.subheader("Data possibility distribution (Hover Original Index)")
    #只抽樣5000
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    df_sample['label_str'] = df_sample['true_label'].map({0: 'Negative (0)', 1:'Positive (1)'})
    fig = px.scatter(
        df_sample,
        x='original_index',
        y = 'probability',
        color = 'label_str',
        hover_data = {
            'original_index': True,
            'probability':':.4f',
            'true_label': True,
            'label_str': False
        },

        labels = {
            'original_index': 'Data Index',
            'probability': 'Predicted Probability',
            'label_str': 'Class'
        },
        title = f'Sample size {sample_size:,} Points Probability Distribution'
    )

    fig.add_hline(
        y = threshold,
        line_dash = 'dash',
        line_color = 'red',
        annotation_text=f'Threshold = {threshold:.3f}',
        annotation_position='top left'
        
    )

    st.plotly_chart(fig, use_container_width=True)
#===========
if "start_analysis" not in st.session_state:
    st.session_state['start_analysis'] = False

def trigger_analysis():
    st.session_state['start_analysis'] = True
#===========================
st.title("System")

with st.container(border=True):
    tab_gc, tab_op = st.tabs(['GC-MD', 'Optical'])

    #======== GC-MD
    with tab_gc:
        st.subheader("GC-MD data upload")

        file_gc = st.file_uploader(
            "Select file",
            key='uploader_gc',
            type=['txt', 'csv']
        )


        if file_gc is not None:
            #讀到之後的preview
            st.success(f"{file_gc} uploaded!")
            try:
                df = pd.read_csv(file_gc)
                st.success(
                    f"Loaded successed! Row:{df.shape[0]}; Columns:{df.shape[1]}"
                )
                st.write('Data Preview')
                st.dataframe(df, use_container_width=True)
                st.button('start analysis', key = 'gc_button', on_click=trigger_analysis)
            except Exception as e:
                st.error(f"file reading failed, please check file format again")
        else:
            st.session_state["start_analysis"] = False


    with tab_op:
        st.subheader("Optical data upload")

        file_op = st.file_uploader(
            "Select file",
            key='uploader_op',
            type=['txt', 'csv']
        )
        st.success(f"{file_op} uploaded!")
        try:
            df = pd.read_csv(file_op)
            st.success(
                f"Loaded successed! Row:{df.shape[0]}; Columns:{df.shape[1]}"
            )
            st.write('Data Preview')
            st.dataframe(df, use_container_width=True)
            if st.button('Start analysis'):
                st.success("See below")
                analysis_results() # 這邊跳轉下面要接的部分
                
        except Exception as e:
            st.error(f"file reading failed, please check file format again")


if st.session_state.get("start_analysis", False):
    st.markdown("---")
    st.header("Analysis Dashboard")
    analysis_results()
#==================== 想辦法接下面，讓他按鈕點了後，會跳下面
