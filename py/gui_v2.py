import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
            except Exception as e:
                st.error(f"file reading failed, please check file format again")
            


    with tab_op:
        st.subheader("Optical data upload")

        file_op = st.file_uploader(
            "Select file",
            key='uploader_op',
            type=['txt', 'csv']
        )
        if file_gc is not None:
            st.success()