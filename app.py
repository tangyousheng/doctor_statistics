import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import os
import plotly.express as px
import plotly.graph_objects as go
import time

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']  # 多种中文字体备选
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_style('whitegrid')
sns.set_palette('pastel')


# 数据处理函数 - 修复日期处理逻辑
def preprocess_data(df):
    # 转换日期格式 - 只对就诊日期做强制要求
    if '就诊日期' in df.columns:
        df['就诊日期'] = pd.to_datetime(df['就诊日期'], errors='coerce')
        # 只删除无效的就诊日期
        df = df.dropna(subset=['就诊日期'])
    else:
        st.warning("数据中缺少 '就诊日期' 列，无法进行时间分析")
        return df
    # 其他日期列允许为空
    for col in ['签约日期', '建档日期']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 创建数值列用于计算
    for col in ['是否本机构建档', '是否外机构建档', '是否本机构签约', '是否外机构签约']:
        if col in df.columns:
            # 处理可能存在的混合类型
            df[col] = df[col].apply(lambda x: 1 if x == '是' or x == 1 else 0)
        else:
            st.error(f"数据中缺少必要列: '{col}'")
    return df


# 计算医生绩效指标 - 修复统计逻辑
def calculate_doctor_performance(df, start_date=None, end_date=None):
    # 筛选日期范围
    if '就诊日期' in df.columns and start_date and end_date:
        mask = (df['就诊日期'] >= pd.Timestamp(start_date)) & (df['就诊日期'] <= pd.Timestamp(end_date))
        df = df.loc[mask]

    # 检查必要列是否存在
    required_columns = ['诊疗医生', '身份证号', '是否本机构建档', '是否外机构建档', '是否本机构签约', '是否外机构签约']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"数据中缺少必要列: {', '.join(missing_cols)}")
        return pd.DataFrame(), None, None

    # 计算每个医生的统计指标 - 修复分组逻辑
    grouped = df.groupby('诊疗医生', as_index=False).agg(
        今日就诊人数=('身份证号', 'count'),
        本机构建档人数=('是否本机构建档', 'sum'),
        外机构建档人数=('是否外机构建档', 'sum'),
        本机构签约人数=('是否本机构签约', 'sum'),
        外机构签约人数=('是否外机构签约', 'sum')
    )

    # 计算未建档人数和未签约人数
    grouped['未建档人数'] = grouped['今日就诊人数'] - grouped['本机构建档人数'] - grouped['外机构建档人数']
    grouped['未签约人数'] = grouped['今日就诊人数'] - grouped['本机构签约人数'] - grouped['外机构签约人数']

    # 计算率（使用今日就诊人数作为分母）
    grouped['建档率'] = grouped['本机构建档人数'] / grouped['今日就诊人数']
    grouped['签约率'] = grouped['本机构签约人数'] / grouped['今日就诊人数']

    # 计算排名
    grouped['建档率排名'] = grouped['建档率'].rank(ascending=False, method='min').astype(int)
    grouped['签约率排名'] = grouped['签约率'].rank(ascending=False, method='min').astype(int)

    # 新建档统计 - 修复逻辑：只统计建档日期等于就诊日期且本机构建档
    new_file_df = None
    if '建档日期' in df.columns and '就诊日期' in df.columns:
        # 创建临时列用于比较日期（忽略时间部分）
        df['建档日期_日期'] = df['建档日期'].dt.date
        df['就诊日期_日期'] = df['就诊日期'].dt.date

        # 筛选新建档记录
        new_file_mask = (df['建档日期_日期'] == df['就诊日期_日期']) & \
                        (df['是否本机构建档'] == 1) & \
                        (df['建档日期'] >= pd.Timestamp(start_date)) & \
                        (df['建档日期'] <= pd.Timestamp(end_date))

        new_file_df = df[new_file_mask].copy()

        # 计算每个医生的新建档人数
        new_file_grouped = new_file_df.groupby('诊疗医生', as_index=False).agg(
            新建档人数=('是否本机构建档', 'sum')
        )

        # 合并到主统计表
        grouped = pd.merge(grouped, new_file_grouped, on='诊疗医生', how='left')
        grouped['新建档人数'] = grouped['新建档人数'].fillna(0).astype(int)

        # 计算新建档率
        grouped['新建档率'] = grouped['新建档人数'] / grouped['今日就诊人数']

        # 计算新建档率排名
        grouped['新建档率排名'] = grouped['新建档率'].rank(ascending=False, method='min').astype(int)

    # 新签约统计
    new_sign_df = None
    if '签约日期' in df.columns:
        # 筛选在统计时间段内签约的记录，且必须是本机构签约
        new_sign_mask = (df['签约日期'] >= pd.Timestamp(start_date)) & \
                        (df['签约日期'] <= pd.Timestamp(end_date)) & \
                        (df['是否本机构签约'] == 1)
        new_sign_df = df[new_sign_mask].copy()

        # 计算每个医生的新签约人数
        new_sign_grouped = new_sign_df.groupby('诊疗医生', as_index=False).agg(
            新签约人数=('是否本机构签约', 'sum')
        )

        # 合并到主统计表
        grouped = pd.merge(grouped, new_sign_grouped, on='诊疗医生', how='left')
        grouped['新签约人数'] = grouped['新签约人数'].fillna(0).astype(int)

        # 计算新签约率
        grouped['新签约率'] = grouped['新签约人数'] / grouped['今日就诊人数']

        # 计算新签约率排名
        grouped['新签约率排名'] = grouped['新签约率'].rank(ascending=False, method='min').astype(int)

    return grouped, new_file_df, new_sign_df

# 生成医生绩效图表 - 统一使用Plotly分组条形图
def generate_performance_charts(performance_df):
    if performance_df.empty:
        return None, None, None, None, None, None

    charts = []

    try:
        # 1. 医生建档和签约数量对比图（使用条形图）
        performance_df = performance_df.sort_values('本机构建档人数', ascending=False)

        # 创建分组条形图
        fig1 = px.bar(
            performance_df,
            x='诊疗医生',
            y=['本机构建档人数', '本机构签约人数'],
            title='医生建档与签约数量对比',
            labels={'value': '人数', 'variable': '类型'},
            barmode='group',
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )

        fig1.update_layout(
            legend_title_text='统计类型',
            xaxis_title='医生姓名',
            yaxis_title='人数',
            hovermode='x unified'
        )
        charts.append(fig1)
    except Exception as e:
        st.error(f"生成对比图时出错: {e}")
        charts.append(None)

    try:
        # 2. 医生建档统计图（包含今日就诊人数、本机构建档人数、外机构建档人数）
        performance_df = performance_df.sort_values('本机构建档人数', ascending=False)

        fig2 = px.bar(
            performance_df,
            x='诊疗医生',
            y=['今日就诊人数', '本机构建档人数', '外机构建档人数', '未建档人数'],
            title='医生建档统计（含今日就诊人数）',
            labels={'value': '人数', 'variable': '类型'},
            barmode='group',
            color_discrete_sequence=['#636EFA', '#00CC96', '#AB63FA', '#FFA15A']
        )

        fig2.update_layout(
            legend_title_text='统计类型',
            xaxis_title='医生姓名',
            yaxis_title='人数',
            hovermode='x unified'
        )
        charts.append(fig2)
    except Exception as e:
        st.error(f"生成建档统计图时出错: {e}")
        charts.append(None)

    try:
        # 3. 医生签约统计图（包含今日就诊人数、本机构签约人数、外机构签约人数）
        performance_df = performance_df.sort_values('本机构签约人数', ascending=False)

        fig3 = px.bar(
            performance_df,
            x='诊疗医生',
            y=['今日就诊人数', '本机构签约人数', '外机构签约人数', '未签约人数'],
            title='医生签约统计（含今日就诊人数）',
            labels={'value': '人数', 'variable': '类型'},
            barmode='group',
            color_discrete_sequence=['#636EFA', '#00CC96', '#AB63FA', '#FFA15A']
        )

        fig3.update_layout(
            legend_title_text='统计类型',
            xaxis_title='医生姓名',
            yaxis_title='人数',
            hovermode='x unified'
        )
        charts.append(fig3)
    except Exception as e:
        st.error(f"生成签约统计图时出错: {e}")
        charts.append(None)

    try:
        # 4. 外机构统计图
        performance_df = performance_df.sort_values('外机构建档人数', ascending=False)
        fig4 = px.bar(performance_df,
                      x='诊疗医生',
                      y=['外机构建档人数', '外机构签约人数'],
                      title='外机构建档与签约统计',
                      labels={'value': '人数', 'variable': '类型'},
                      barmode='group',
                      color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        fig4.update_layout(
            title='外机构建档与签约统计',
            legend_title_text='统计类型',
            xaxis_title='医生姓名',
            yaxis_title='人数',
            hovermode='x unified'
        )
        charts.append(fig4)
    except Exception as e:
        st.error(f"生成外机构统计图时出错: {e}")
        charts.append(None)

    try:
        # 5. 新签约统计图（包含今日就诊人数和新签约人数）
        if '新签约人数' in performance_df.columns:
            # 按新签约人数降序排列
            performance_df = performance_df.sort_values('新签约人数', ascending=False)

            fig5 = px.bar(
                performance_df,
                x='诊疗医生',
                y=['今日就诊人数', '新签约人数'],
                title='医生新签约统计（含今日就诊人数）',
                labels={'value': '人数', 'variable': '类型'},
                barmode='group',
                color_discrete_sequence=['#636EFA', '#EF553B']
            )

            # 添加新签约率作为折线图（次坐标轴）
            fig5.add_trace(
                go.Scatter(
                    x=performance_df['诊疗医生'],
                    y=performance_df['新签约率'],
                    name='新签约率',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color='#FFA15A', width=3),
                    marker=dict(size=10, symbol='diamond')
                )
            )

            fig5.update_layout(
                legend_title_text='统计类型',
                xaxis_title='医生姓名',
                yaxis_title='人数',
                yaxis2=dict(
                    title='新签约率',
                    overlaying='y',
                    side='right',
                    tickformat='.0%',
                    range=[0, max(performance_df['新签约率']) * 1.2]  # 设置范围，留出空间
                ),
                hovermode='x unified'
            )

            # 添加排名信息到悬停文本
            fig5.update_traces(
                hovertemplate="<br>".join([
                    "医生: %{x}",
                    "值: %{y}",
                    "<extra></extra>"
                ])
            )

            # 为折线图添加单独的悬停文本
            fig5.update_traces(
                selector=dict(name='新签约率'),
                hovertemplate="<br>".join([
                    "医生: %{x}",
                    "新签约率: %{y:.2%}",
                    "排名: %{customdata}",
                    "<extra></extra>"
                ]),
                customdata=performance_df['新签约率排名']
            )

            charts.append(fig5)
    except Exception as e:
        st.error(f"生成新签约统计图时出错: {e}")
        charts.append(None)

    try:
        # 6. 新建档统计图（包含今日就诊人数和新建档人数）
        if '新建档人数' in performance_df.columns:
            # 按新建档人数降序排列
            performance_df = performance_df.sort_values('新建档人数', ascending=False)

            fig6 = px.bar(
                performance_df,
                x='诊疗医生',
                y=['今日就诊人数', '新建档人数'],
                title='医生新建档统计（含今日就诊人数）',
                labels={'value': '人数', 'variable': '类型'},
                barmode='group',
                color_discrete_sequence=['#636EFA', '#19D3F3']
            )

            # 添加新建档率作为折线图（次坐标轴）
            fig6.add_trace(
                go.Scatter(
                    x=performance_df['诊疗医生'],
                    y=performance_df['新建档率'],
                    name='新建档率',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color='#FF6692', width=3),
                    marker=dict(size=10, symbol='diamond')
                )
            )

            fig6.update_layout(
                legend_title_text='统计类型',
                xaxis_title='医生姓名',
                yaxis_title='人数',
                yaxis2=dict(
                    title='新建档率',
                    overlaying='y',
                    side='right',
                    tickformat='.0%',
                    range=[0, max(performance_df['新建档率']) * 1.2]  # 设置范围，留出空间
                ),
                hovermode='x unified'
            )

            # 添加排名信息到悬停文本
            fig6.update_traces(
                hovertemplate="<br>".join([
                    "医生: %{x}",
                    "值: %{y}",
                    "<extra></extra>"
                ])
            )

            # 为折线图添加单独的悬停文本
            fig6.update_traces(
                selector=dict(name='新建档率'),
                hovertemplate="<br>".join([
                    "医生: %{x}",
                    "新建档率: %{y:.2%}",
                    "排名: %{customdata}",
                    "<extra></extra>"
                ]),
                customdata=performance_df['新建档率排名']
            )

            charts.append(fig6)
    except Exception as e:
        st.error(f"生成新建档统计图时出错: {e}")
        charts.append(None)

    return charts


# 主应用
def main():
    # 设置页面配置
    st.set_page_config(
        page_title="医生签约建档统计系统",
        layout="wide",
        page_icon="🏥",
        initial_sidebar_state="expanded"
    )

    # 自定义CSS样式
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stSelectbox, .stDateInput {
        max-width: 300px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #212529;
    }
    .header {
        background-color: #1e88e5;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 页面标题
    st.markdown('<div class="header"><h1>🏥 医生签约建档统计系统</h1></div>', unsafe_allow_html=True)

    # 上传数据
    st.subheader("📤 数据上传")
    uploaded_file = st.file_uploader("上传数据文件 (CSV/Excel)", type=["csv", "xlsx"], label_visibility="collapsed")

    # 使用状态管理
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'performance_df' not in st.session_state:
        st.session_state.performance_df = None
    if 'new_sign_list' not in st.session_state:
        st.session_state.new_sign_list = None
    if 'new_file_list' not in st.session_state:
        st.session_state.new_file_list = None

    if uploaded_file is not None:
        try:
            # 读取数据
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            if file_ext == '.csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # 保存到session state
            st.session_state.df = df

            # 显示原始数据
            with st.expander("📋 查看原始数据(前20条)"):
                st.dataframe(df.head(20))

            # 预处理数据
            df_processed = preprocess_data(df.copy())

            # 日期范围选择
            if '就诊日期' in df_processed.columns:
                min_date = df_processed['就诊日期'].min().to_pydatetime().date()
                max_date = df_processed['就诊日期'].max().to_pydatetime().date()
            else:
                min_date = datetime.today().date() - timedelta(days=30)
                max_date = datetime.today().date()

            st.subheader("📅 选择统计时间范围")
            col1, col2 = st.columns(2)
            start_date = col1.date_input("开始日期", min_date, min_value=min_date, max_value=max_date)
            end_date = col2.date_input("结束日期", max_date, min_value=min_date, max_value=max_date)

            # 计算绩效
            if st.button("计算绩效指标", key="calculate_perf"):
                with st.spinner('正在计算绩效指标...'):
                    start_time = time.time()
                    performance_df, new_file_list, new_sign_list = calculate_doctor_performance(df_processed,
                                                                                                start_date, end_date)
                    st.session_state.performance_df = performance_df
                    st.session_state.new_file_list = new_file_list
                    st.session_state.new_sign_list = new_sign_list

                    end_time = time.time()
                    st.success(f"计算完成! 耗时: {end_time - start_time:.2f}秒")

            # 显示绩效数据
            if st.session_state.performance_df is not None and not st.session_state.performance_df.empty:
                performance_df = st.session_state.performance_df

                with st.expander("📊 查看医生绩效统计"):
                    # 格式化百分比列
                    formatted_df = performance_df.copy()
                    formatted_df['建档率'] = formatted_df['建档率'].apply(lambda x: f"{x:.2%}")
                    formatted_df['签约率'] = formatted_df['签约率'].apply(lambda x: f"{x:.2%}")

                    # 如果有新建档率列，也格式化
                    if '新建档率' in formatted_df.columns:
                        formatted_df['新建档率'] = formatted_df['新建档率'].apply(lambda x: f"{x:.2%}")
                        # 确保新建档率排名列存在
                        if '新建档率排名' not in formatted_df.columns:
                            formatted_df['新建档率排名'] = 'N/A'

                    # 如果有新签约率列，也格式化
                    if '新签约率' in formatted_df.columns:
                        formatted_df['新签约率'] = formatted_df['新签约率'].apply(lambda x: f"{x:.2%}")
                        # 确保新签约率排名列存在
                        if '新签约率排名' not in formatted_df.columns:
                            formatted_df['新签约率排名'] = 'N/A'

                    # 显示表格
                    st.dataframe(formatted_df)

                    # 关键指标摘要
                    st.subheader("📌 关键指标摘要")
                    col1, col2, col3, col4 = st.columns(4)
                    # 计算总计
                    total_visits = performance_df['今日就诊人数'].sum()
                    total_local_files = performance_df['本机构建档人数'].sum()
                    total_local_signs = performance_df['本机构签约人数'].sum()
                    total_external_files = performance_df['外机构建档人数'].sum()
                    total_external_signs = performance_df['外机构签约人数'].sum()
                    # 新增：未建档和未签约人数
                    total_unfiled = performance_df['未建档人数'].sum()
                    total_unsigned = performance_df['未签约人数'].sum()
                    # 添加今日就诊人数指标卡（第一行第一个）
                    col1.markdown(
                        '<div class="metric-card"><div class="metric-title">今日就诊人数</div><div class="metric-value">{}</div></div>'.format(
                            total_visits),
                        unsafe_allow_html=True)
                    col2.markdown(
                        '<div class="metric-card"><div class="metric-title">本机构建档率</div><div class="metric-value">{:.2%}</div></div>'.format(
                            total_local_files / total_visits),
                        unsafe_allow_html=True)
                    col3.markdown(
                        '<div class="metric-card"><div class="metric-title">本机构签约率</div><div class="metric-value">{:.2%}</div></div>'.format(
                            total_local_signs / total_visits),
                        unsafe_allow_html=True)
                    col4.markdown(
                        '<div class="metric-card"><div class="metric-title">外机构建档人数</div><div class="metric-value">{}</div></div>'.format(
                            total_external_files),
                        unsafe_allow_html=True)
                    # 第二行指标
                    col1, col2, col3, col4 = st.columns(4)
                    col1.markdown(
                        '<div class="metric-card"><div class="metric-title">外机构签约人数</div><div class="metric-value">{}</div></div>'.format(
                            total_external_signs),
                        unsafe_allow_html=True)
                    col2.markdown(
                        '<div class="metric-card"><div class="metric-title">可直接建档人数</div><div class="metric-value">{}</div></div>'.format(
                            total_unfiled),
                        unsafe_allow_html=True)
                    col3.markdown(
                        '<div class="metric-card"><div class="metric-title">可直接签约人数</div><div class="metric-value">{}</div></div>'.format(
                            total_unsigned),
                        unsafe_allow_html=True)
                    # 如果有新建档人数，显示相关指标
                    if '新建档人数' in performance_df.columns:
                        total_new_files = performance_df['新建档人数'].sum()
                        col4.markdown(
                            '<div class="metric-card"><div class="metric-title">当日新建档人数</div><div class="metric-value">{}</div></div>'.format(
                                total_new_files),
                            unsafe_allow_html=True)
                    # 第三行指标（如果有新签约人数）
                    if '新签约人数' in performance_df.columns:
                        total_new_signs = performance_df['新签约人数'].sum()
                        col1, col2, col3, col4 = st.columns(4)
                        col1.markdown(
                            '<div class="metric-card"><div class="metric-title">当日新签约人数</div><div class="metric-value">{}</div></div>'.format(
                                total_new_signs),
                            unsafe_allow_html=True)
                        col2.markdown(
                            '<div class="metric-card"><div class="metric-title">当日新签约率</div><div class="metric-value">{:.2%}</div></div>'.format(
                                total_new_signs / total_visits),
                            unsafe_allow_html=True)

                # 生成图表
                st.subheader("📈 绩效可视化")
                charts = generate_performance_charts(performance_df)

                # 显示所有图表
                for chart in charts:
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                # 显示新建档名单
                if st.session_state.new_file_list is not None and not st.session_state.new_file_list.empty:
                    st.subheader("📝 新建档名单")
                    st.write(
                        f"在 {start_date} 至 {end_date} 期间新建档的患者列表 (共{len(st.session_state.new_file_list)}人)：")
                    st.dataframe(st.session_state.new_file_list[['诊疗医生', '身份证号', '建档日期', '就诊日期']])

                # 显示新签约名单
                if st.session_state.new_sign_list is not None and not st.session_state.new_sign_list.empty:
                    st.subheader("📝 新签约名单")
                    st.write(
                        f"在 {start_date} 至 {end_date} 期间新签约的患者列表 (共{len(st.session_state.new_sign_list)}人)：")
                    st.dataframe(st.session_state.new_sign_list[['诊疗医生', '身份证号', '签约日期']])

                # 数据导出
                st.subheader("💾 数据导出")
                col1, col2 = st.columns(2)

                # 导出绩效数据
                output = io.BytesIO()
                try:
                    # 创建Excel写入器
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        performance_df.to_excel(writer, sheet_name='绩效统计', index=False)
                        st.session_state.df.to_excel(writer, sheet_name='原始数据', index=False)

                        # 如果有新建档名单，也导出
                        if st.session_state.new_file_list is not None:
                            st.session_state.new_file_list.to_excel(writer, sheet_name='新建档名单', index=False)

                        # 如果有新签约名单，也导出
                        if st.session_state.new_sign_list is not None:
                            st.session_state.new_sign_list.to_excel(writer, sheet_name='新签约名单', index=False)
                except Exception as e:
                    st.error(f"导出Excel时出错: {e}")
                output.seek(0)

                col1.download_button(
                    label="下载绩效统计报告 (Excel)",
                    data=output,
                    file_name=f"医生绩效统计_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # 导出图表按钮（占位）
                col2.download_button(
                    label="下载所有图表 (PDF)",
                    data=output,
                    file_name=f"医生绩效图表_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    disabled=True  # 实际应用中需要实现PDF导出功能
                )

        except Exception as e:
            st.error(f"数据处理错误: {str(e)}")
            st.exception(e)
    else:
        # 仅显示上传提示
        st.info("请上传CSV或Excel格式的数据文件以开始分析")


if __name__ == "__main__":
    main()
