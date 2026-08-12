import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
from scipy import stats
import warnings

# kaleido 检测
try:
    import plotly.io as pio
    _ = pio.kaleido.scope
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------- 页面设置 ----------
st.set_page_config(page_title="轻量数据分析工具", layout="wide")
st.title("📊 轻量数据分析工具")

# ---------- 文件上传 ----------
uploaded_file = st.file_uploader("1. 上传数据文件", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 清洗对象列
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].fillna("").astype(str).astype("string")

    st.success(f"已加载数据，共 {df.shape[0]} 行，{df.shape[1]} 列")

    # 统计卡片
    total_rows, total_cols = df.shape
    num_cols_count = len(df.select_dtypes(include="number").columns)
    total_missing = df.isnull().sum().sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 总行数", total_rows)
    col2.metric("📁 总列数", total_cols)
    col3.metric("🔢 数值列", num_cols_count)
    col4.metric("❓ 缺失值总数", total_missing)
    st.markdown("---")

    # ---------- 侧边栏过滤 ----------
    st.sidebar.header("2. 数据过滤")
    all_columns = df.columns.tolist()
    filter_column = st.sidebar.selectbox("选择过滤字段", ["无"] + all_columns)

    if filter_column != "无":
        if df[filter_column].dtype == "string" or df[filter_column].nunique() < 20:
            unique_vals = df[filter_column].dropna().unique().tolist()
            selected_vals = st.sidebar.multiselect(
                f"选择 {filter_column} 的值",
                options=unique_vals,
                default=unique_vals
            )
            if selected_vals:
                df = df[df[filter_column].isin(selected_vals)]
        else:
            min_val, max_val = float(df[filter_column].min()), float(df[filter_column].max())
            selected_range = st.sidebar.slider(
                f"{filter_column} 范围",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            df = df[(df[filter_column] >= selected_range[0]) & (df[filter_column] <= selected_range[1])]

    # ---------- 主区域：4个标签页 ----------
    tab1, tab2, tab3, tab4 = st.tabs(["📋 数据表", "📈 可视化", "📊 分组分析", "🔬 高级分析"])

    # ==================== 标签页1：数据表 ====================
    with tab1:
        st.subheader("过滤后的数据预览")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="下载当前数据为 CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    # ==================== 标签页2：可视化 ====================
    with tab2:
        st.subheader("交互式图表")
        chart_type = st.radio("图表类型",
            ["散点图", "柱状图", "折线图", "面积图", "箱线图", "直方图", "饼图", "热力图"],
            horizontal=True
        )

        if chart_type == "饼图":
            cat_col = st.selectbox("分类列", all_columns)
            num_col = st.selectbox("数值列", df.select_dtypes(include="number").columns)
            fig = px.pie(df, names=cat_col, values=num_col, title=f"{cat_col} 分布")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "热力图":
            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                st.warning("至少需要两个数值列才能绘制热力图")
            else:
                corr = num_df.corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="数值列相关性热力图")
                st.plotly_chart(fig, use_container_width=True)

        else:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X 轴", all_columns)
            with col2:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols and chart_type != "直方图":
                    y_axis = st.selectbox("Y 轴", num_cols)
                else:
                    y_axis = None

            if (y_axis is None and chart_type not in ["直方图", "箱线图"]) or (chart_type == "箱线图" and y_axis is None):
                st.info("请选择合适的 X/Y 轴")
            else:
                if chart_type == "散点图":
                    color_col = st.selectbox("颜色分组（可选）", ["无"] + all_columns)
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=None if color_col == "无" else color_col)
                elif chart_type == "柱状图":
                    fig = px.bar(df, x=x_axis, y=y_axis)
                elif chart_type == "折线图":
                    fig = px.line(df, x=x_axis, y=y_axis)
                elif chart_type == "面积图":
                    fig = px.area(df, x=x_axis, y=y_axis)
                elif chart_type == "箱线图":
                    if y_axis:
                        fig = px.box(df, x=x_axis, y=y_axis)
                    else:
                        fig = px.box(df, x=x_axis)
                else:  # 直方图
                    fig = px.histogram(df, x=x_axis)

                st.plotly_chart(fig, use_container_width=True)

    # ==================== 标签页3：分组分析 ====================
    with tab3:
        st.subheader("分组聚合分析")
        group_col = st.selectbox("分组字段", all_columns)
        num_cols_agg = df.select_dtypes(include="number").columns.tolist()
        if num_cols_agg:
            agg_col = st.selectbox("聚合字段（数值）", num_cols_agg)
            agg_func = st.selectbox("聚合方式", ["平均值", "总和", "计数", "最大值", "最小值"])
            agg_map = {"平均值": "mean", "总和": "sum", "计数": "count", "最大值": "max", "最小值": "min"}
            result = df.groupby(group_col, as_index=False).agg({agg_col: agg_map[agg_func]})
            st.dataframe(result, use_container_width=True)
            fig_bar = px.bar(result, x=group_col, y=agg_col, title=f"{group_col} 的 {agg_func}")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("数据中没有数值列，无法进行聚合分析")

    # ==================== 标签页4：高级分析 ====================
    with tab4:
        st.subheader("🔬 高级数据分析")
        analysis_type = st.radio("选择分析类型",
            ["相关性洞察", "异常值检测", "缺失值处理"],
            horizontal=True
        )

        num_df = df.select_dtypes(include="number")
        if num_df.empty and analysis_type != "缺失值处理":
            st.warning("当前数据没有数值列，部分分析功能不可用")
        else:
            # ---------- 相关性洞察 ----------
            if analysis_type == "相关性洞察":
                st.markdown("### 🔗 数值变量相关性")
                if num_df.shape[1] < 2:
                    st.warning("需要至少两个数值列")
                else:
                    corr = num_df.corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto",
                                    title="相关性矩阵", color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig, use_container_width=True)

                    # 找出最强相关对
                    corr_pairs = corr.unstack().reset_index()
                    corr_pairs.columns = ["变量1", "变量2", "相关系数"]
                    corr_pairs = corr_pairs[corr_pairs["变量1"] < corr_pairs["变量2"]]
                    corr_pairs["abs_corr"] = corr_pairs["相关系数"].abs()
                    corr_pairs = corr_pairs.sort_values("abs_corr", ascending=False)

                    st.markdown("#### 相关性最强的前5对变量")
                    top5 = corr_pairs.head(5)
                    for i, row in top5.iterrows():
                        var1, var2, corr_val = row["变量1"], row["变量2"], row["相关系数"]
                        st.write(f"**{var1}** 与 **{var2}**：相关系数 = {corr_val:.3f}")
                        # 小散点图预览
                        fig_pair = px.scatter(df, x=var1, y=var2, title=f"{var1} vs {var2}")
                        st.plotly_chart(fig_pair, use_container_width=True)

            # ---------- 异常值检测 ----------
            elif analysis_type == "异常值检测":
                st.markdown("### 🚨 异常值检测 (Z-Score 方法)")
                num_col = st.selectbox("选择要检测的数值列", num_df.columns)
                threshold = st.slider("Z-Score 阈值（通常为 2.5 ~ 3.5）", 1.5, 5.0, 3.0, 0.1)

                if num_col:
                    col_data = df[num_col].dropna()
                    z_scores = np.abs(stats.zscore(col_data))
                    outliers_mask = z_scores > threshold
                    outliers_count = outliers_mask.sum()
                    total_count = len(col_data)

                    st.metric("异常值数量", f"{outliers_count} / {total_count}",
                              f"{outliers_count/total_count*100:.1f}%")

                    if outliers_count > 0:
                        # 箱线图显示异常值
                        fig_box = px.box(df, y=num_col, title=f"{num_col} 分布与异常值")
                        st.plotly_chart(fig_box, use_container_width=True)

                        # 展示异常数据行
                        st.markdown("#### 异常数据明细（基于当前过滤后数据）")
                        outliers_df = df[outliers_mask]
                        st.dataframe(outliers_df, use_container_width=True)
                    else:
                        st.success("未检测到明显异常值")

            # ---------- 缺失值处理 ----------
            else:
                st.markdown("### 💧 缺失值概况")
                missing = df.isnull().sum()
                missing = missing[missing > 0].sort_values(ascending=False)
                if missing.empty:
                    st.success("数据没有缺失值！")
                else:
                    st.dataframe(missing.rename("缺失数量"), use_container_width=True)

                    st.markdown("### 处理缺失值")
                    col_to_fix = st.selectbox("选择要处理的列", missing.index.tolist())
                    strategy = st.radio("处理策略",
                        ["删除缺失值所在行", "用 0 填充", "用均值填充（仅数值）", "用众数填充"],
                        horizontal=True)

                    if st.button("应用处理"):
                        if strategy == "删除缺失值所在行":
                            df.dropna(subset=[col_to_fix], inplace=True)
                        elif strategy == "用 0 填充":
                            df[col_to_fix].fillna(0, inplace=True)
                        elif strategy == "用均值填充（仅数值）":
                            if col_to_fix in num_df.columns:
                                df[col_to_fix].fillna(df[col_to_fix].mean(), inplace=True)
                            else:
                                st.error("该列不是数值列，无法使用均值填充")
                        elif strategy == "用众数填充":
                            mode_val = df[col_to_fix].mode()
                            if not mode_val.empty:
                                df[col_to_fix].fillna(mode_val[0], inplace=True)
                            else:
                                st.warning("无法计算众数")
                        st.success(f"已对 '{col_to_fix}' 执行 {strategy}，数据已更新。请在其他标签页查看最新结果。")

    # ==================== 导出报告 ====================
    st.sidebar.markdown("---")
    st.sidebar.header("📥 导出分析报告")

    from openpyxl.drawing.image import Image as XLImage

    def insert_plotly_fig_openpyxl(fig, sheet, cell):
        img_bytes = fig.to_image(format="png", scale=2)
        img_stream = BytesIO(img_bytes)
        img = XLImage(img_stream)
        img.width, img.height = 600, 400
        sheet.add_image(img, cell)

    towrite = BytesIO()
    with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="数据", index=False)
        workbook = writer.book
        worksheet = writer.sheets["数据"]

        if KALEIDO_AVAILABLE:
            chart_sheet = workbook.create_sheet("图表")
            row_cell = "A1"

            num_cols_list = df.select_dtypes(include="number").columns.tolist()
            if num_cols_list:
                fig_hist = px.histogram(df, x=num_cols_list[0], title=f"{num_cols_list[0]} 分布")
                insert_plotly_fig_openpyxl(fig_hist, chart_sheet, row_cell)
                row_cell = "A21"

            cat_cols = df.select_dtypes(include="string").columns.tolist()
            if cat_cols:
                cat_col = cat_cols[0]
                counts = df[cat_col].value_counts().reset_index()
                counts.columns = [cat_col, "计数"]
                fig_pie = px.pie(counts, names=cat_col, values="计数", title=f"{cat_col} 占比")
                insert_plotly_fig_openpyxl(fig_pie, chart_sheet, row_cell)
                row_cell = "A41"

            if len(num_cols_list) >= 2:
                corr = df[num_cols_list].corr()
                fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="相关性热力图")
                insert_plotly_fig_openpyxl(fig_heat, chart_sheet, row_cell)

    towrite.seek(0)
    st.sidebar.download_button(
        label="下载完整报告 (Excel)",
        data=towrite,
        file_name="数据分析报告.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )