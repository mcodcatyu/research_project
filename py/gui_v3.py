import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

st.set_page_config(layout='wide')
st.title("threshold and numbers")
st.write('try difffernt threshold')

#==========
def load_data():
    np.random.seed(42)
    n_neg, n_pos=139717, 15729
    y_neg = np.zeros(n_neg)
    y_pos = np.ones(n_pos)

    prob_neg = np.random.beta(1, 15, n_neg)
    prob_pos = np.random.beta(4, 4, n_pos)

    y_true = np.concatenate([y_neg, y_pos])
    y_prob = np.concatenate([prob_neg, prob_pos])
    return y_true, y_prob


y_true, y_prob = load_data()

precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)


#========== Sidebar
st.sidebar.header("Control screeen")

recommmended_thres = 0.4694
st.sidebar.info(f"Recommend best threshold:{recommmended_thres}")

threshold = st.sidebar.slider(
    "Adjust threshold",
    min_value = 0.000,
    max_value = 1.000,
    value=recommmended_thres,
    step=0.001
)

y_proba = (y_prob >=threshold).astype(int)

total_samples = len(y_true)
selected_count = int(np.sum(y_proba))
selected_ratio = (selected_count / total_samples) * 100

idx = np.argmin(np.abs(thresholds- threshold))
current_prec = precisions[idx]
current_rec= recalls[idx]
current_f1 = 2 * (current_prec * current_rec) / (current_prec + current_rec + 1e-7)


col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label='selected positive labels', value=f"{selected_count}", delta=f"%in total {selected_ratio:.4f}%")
with col2:
    st.metric(label='Current Precision', value=f"{current_prec:.4f}")
with col3:
    st.metric(label='Recall / F1-score', value=f"{current_rec:.4f}", delta=f"F1:{current_f1:.4f}")

st.markdown("----------")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Bar Chart")

    metrics_df = pd.DataFrame({
        'metric': ['Precision', 'Recall', 'F1-score'],
        'value': [current_prec, current_rec, current_f1]
    })
    st.bar_chart(data=metrics_df, x='metric', y='value')

with col_right:
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