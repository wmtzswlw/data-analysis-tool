import streamlit as st
import pandas as pd

st.set_page_config(page_title="轻量数据分析工具", layout="wide")
st.title("📊 我的数据分析工具")

uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("数据预览")
    st.dataframe(df, use_container_width=True)

    st.subheader("基本统计信息")
    st.write(df.describe(include="all"))
