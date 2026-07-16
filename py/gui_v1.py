import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

"""
# My first app
Here's our first attempt at using data to create a table:
"""

def retrain_model(df):

    X = df[features]
    y = df['label1']
    # 這邊放邏輯

    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    st.success("Model retrain completed!")
if uploaded_file is not None:
    model_path = ""
    df = pd.read_csv(uploaded_file)

features = # 自己寫 這邊feature定義

#============ streamlit gui 
model = joblib.load(model_path) # 定義模型

st.
#===========
st.title("Simple Data Dashboard")
st.markdown(""" This GUI is to help detect anomaly data
            """)
uploaded_file = st.file_uploader("Choose a CSV file", type='csv') # file choose

if uploaded_file is not None:
    st.write("File uploaded...")

    df = pd.read_csv(uploaded_file) # 這裡要做一些轉換(因為他原本用的是txt檔案)

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Filter Data")
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select column to filter by", columns)
    unique_values = df[selected_column].unique()
    selected_value = st.selectbox("Select value", unique_values)

    filtered_df = df[df[selected_column] == selected_value]
    st.write(filtered_df)

    st.subheader("Plot Data")
    x_column = st.selectbox("Select x-axis column", columns)
    y_column = st.selectbox("Select y-axis column", columns)

    if st.buton("Generate Plot"):
        st.line_chart(filtered_df.set_index(x_column)[y_column])
else: 
    st.write("Waiting on file upload...")

