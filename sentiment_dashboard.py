#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
茶饮/咖啡品牌舆情风险分析 Dashboard
适配数据格式：产品评价_茶饮舆论.xlsx
使用 Streamlit + Plotly 构建
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import re
from io import BytesIO

# 设置页面配置
st.set_page_config(
    page_title="茶饮/咖啡品牌舆情风险分析 Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式配置 ====================
def set_page_style():
    """设置页面样式"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 配色方案
COLOR_SCHEME = {
    'positive': '#2ca02c',
    'neutral': '#ffbb78',
    'negative': '#d62728',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e'
}

SENTIMENT_COLORS = {
    '正向': '#2ca02c',
    '中性': '#ffbb78',
    '负向': '#d62728'
}

# ==================== 风险关键词词典 ====================
RISK_KEYWORDS = {
    '口味': ['甜', '苦', '酸', '淡', '浓', '腻', '香精', '糖', '味道', '好喝', '难喝', '怪味', '异味',
            '口感', '冰', '凉', '热', '温', '稀', '稠', '顺滑', '粗糙', '颗粒', '沉淀'],
    '食安': ['细菌', '病毒', '霉菌', '发霉', '变质', '腐烂', '臭', '长毛', '虫子', '虫卵', '污染', 
             '农药', '残留', '重金属', '铅', '汞', '铬', '塑化剂', '有毒', '毒',
             '添加物', '添加剂', '香料', '色素', '防腐剂', '甜味剂', '人工', '合成',
             '不清楚', '成分不明', '标签不清', '成分不详', '配方保密', '信息不足', '不透明',
             '卫生', '清洁', '消毒', '脏', '脏兮兮', '不卫生', '交叉污染', '冷链', '温度', '食物中毒', '拉肚子', '腹泻']
}

RISK_TYPE_COLORS = {
    '口味': '#e74c3c',
    '食安': '#c0392b',
    '其他': '#95a5a6'
}

# ==================== 数据加载与清洗 ====================

@st.cache_data
def load_and_process_data(file_path=None):
    """加载并处理数据"""
    try:
        if file_path:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        else:
            return None
        
        return clean_and_enrich_data(df)
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

def clean_and_enrich_data(df):
    """
    数据清洗与丰富：
    - 标准化列名
    - 转换时间格式
    - 从情感向映射风险等级
    - 从评论内容提取风险类型
    """
    df = df.copy()
    
    # 列名映射（中文->英文）
    column_mapping = {
        '品牌': 'brand',
        '产品': 'product',
        '评论内容': 'comment',
        '评论日期': 'datetime',
        'IP 属地': 'region',
        '情感向': 'sentiment',
        '用户名': 'username',
        '是否二上': 'is_second',
        '出上线时间': 'launch_date',
        '记录日期': 'record_date'
    }
    
    # 重命名存在的列
    for old, new in column_mapping.items():
        if old in df.columns:
            df[new] = df[old]
    
    # 处理评论日期
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # 如果有缺失的时间，尝试用记录日期
    if df['datetime'].isna().any() and 'record_date' in df.columns:
        df['datetime'] = df['datetime'].fillna(pd.to_datetime(df['record_date'], errors='coerce'))
    
    # 填充仍缺失的时间
    df['datetime'].fillna(pd.Timestamp.now(), inplace=True)
    
    # 提取日期字段
    df['date'] = df['datetime'].dt.date
    df['year_month'] = df['datetime'].dt.to_period('M').astype(str)
    df['week'] = df['datetime'].dt.to_period('W').astype(str)
    
    # 标准化情感向
    sentiment_mapping = {
        '正向': '正向',
        '中性': '中性',
        '负向': '负向'
    }
    df['sentiment'] = df['sentiment'].map(sentiment_mapping).fillna('中性')
    
    # 将情感向映射为风险等级
    risk_mapping = {
        '正向': '低',
        '中性': '中',
        '负向': '高'
    }
    df['risk_level'] = df['sentiment'].map(risk_mapping)
    
    # 从评论内容提取风险类型
    df['risk_type'] = df['comment'].apply(extract_risk_type)
    
    # 处理缺失值
    df['brand'] = df['brand'].fillna('未知品牌')
    df['product'] = df['product'].fillna('未知产品')
    df['comment'] = df['comment'].fillna('')
    df['region'] = df.get('region', pd.Series(['未知'] * len(df))).fillna('未知')
    
    # 添加风险标记（负向和中性视为风险）
    df['is_risk'] = df['sentiment'].isin(['负向', '中性'])
    
    return df

def extract_risk_type(comment):
    """从评论内容提取风险类型"""
    if pd.isna(comment):
        return '其他'
    
    comment = str(comment)
    scores = {}
    
    for risk_type, keywords in RISK_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in comment)
        if score > 0:
            scores[risk_type] = score
    
    if scores:
        return max(scores, key=scores.get)
    return '其他'

# ==================== 筛选器组件 ====================

def render_filters(df):
    """渲染侧边栏筛选器"""
    st.sidebar.header("🔍 数据筛选")
    
    # 品牌多选
    brands = sorted(df['brand'].unique())
    selected_brands = st.sidebar.multiselect(
        "选择品牌",
        options=brands,
        default=brands
    )
    
    # 产品多选（可选，避免太多选项）
    products = sorted(df['product'].unique())
    selected_products = st.sidebar.multiselect(
        "选择产品",
        options=products,
        default=products
    )
    
    # 时间范围选择
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    # 默认选中本周（周一到今天）
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())  # 本周一
    default_start = max(week_start, min_date)  # 不早于数据最早日期
    default_end = min(today, max_date)         # 不晚于数据最晚日期
    
    date_range = st.sidebar.date_input(
        "选择时间范围",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date
    )
    
    # 情感向筛选
    sentiments = ['正向', '中性', '负向']
    selected_sentiments = st.sidebar.multiselect(
        "情感向",
        options=sentiments,
        default=sentiments
    )
    
    # 风险类型筛选
    risk_types = sorted(df['risk_type'].unique())
    selected_risk_types = st.sidebar.multiselect(
        "风险类型",
        options=risk_types,
        default=risk_types
    )
    
    # 应用筛选
    filtered_df = df[
        (df['brand'].isin(selected_brands)) &
        (df['product'].isin(selected_products)) &
        (df['sentiment'].isin(selected_sentiments)) &
        (df['risk_type'].isin(selected_risk_types))
    ]
    
    # 时间筛选
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= date_range[0]) &
            (filtered_df['date'] <= date_range[1])
        ]
    
    # 重置按钮
    if st.sidebar.button("🔄 重置筛选"):
        st.rerun()
    
    return filtered_df

# ==================== 可视化模块 ====================

def render_overview_metrics(df):
    """渲染数据总览指标卡"""
    st.markdown('<div class="main-header">☕ 茶饮/咖啡品牌舆情风险分析 Dashboard</div>', unsafe_allow_html=True)
    
    # ==================== 产品风险等级排行榜 ====================
    st.subheader("🚨 产品风险等级排行榜")
    
    # 计算每个产品的风险指标
    product_metrics = []
    
    for product in df['product'].unique():
        product_data = df[df['product'] == product]
        total = len(product_data)
        high_risk = len(product_data[product_data['risk_level'] == '高'])
        medium_risk = len(product_data[product_data['risk_level'] == '中'])
        low_risk = len(product_data[product_data['risk_level'] == '低'])
        
        # 获取产品所属品牌（取出现最多的品牌）
        brand = product_data['brand'].mode()[0] if len(product_data['brand'].mode()) > 0 else '未知品牌'
        
        # 风险评分 = 负向评论数 / 总评论数（百分比，0-100）
        risk_score = (high_risk / total * 100) if total > 0 else 0
        
        product_metrics.append({
            'product': product,
            'brand': brand,
            'total': total,
            'high': high_risk,
            'medium': medium_risk,
            'low': low_risk,
            'risk_score': risk_score,
        })
    
    product_metrics_df = pd.DataFrame(product_metrics).sort_values('risk_score', ascending=False)
    
    # 显示排行榜（取Top 10）
    top_n = 10
    cols = st.columns(5)  # 每行5个产品
    
    for idx, (_, row) in enumerate(product_metrics_df.head(top_n).iterrows()):
        score = row['risk_score']
        if score >= 50:
            color = '#d62728'
            level = '🔴'
        elif score >= 25:
            color = '#ffbb78'
            level = '🟡'
        else:
            color = '#2ca02c'
            level = '🟢'
        
        col_idx = idx % 5
        with cols[col_idx]:
            st.markdown(
                f"<div style='padding: 12px; margin: 4px 0; border-radius: 8px; "
                f"background: linear-gradient(135deg, {color}20 0%, {color}05 100%); "
                f"border-left: 4px solid {color};'>"
                f"<div style='font-weight: bold; margin-bottom: 4px;'>{level} {row['product'][:12]}</div>"
                f"<div style='font-size: 11px; color: #555; margin-bottom: 6px;'>"
                f"品牌: <b>{row['brand']}</b>"
                f"</div>"
                f"<div style='font-size: 12px; color: #666;'>"
                f"负向占比: <b style='color: {color};'>{row['risk_score']:.1f}%</b><br/>"
                f"负向{row['high']} 中性{row['medium']} 正向{row['low']}<br/>"
                f"<span style='font-size: 10px; color: #999;'>评论数: {row['total']}</span>"
                f"</div></div>",
                unsafe_allow_html=True
            )
        
        if (idx + 1) % 5 == 0 and idx < top_n - 1:
            cols = st.columns(5)
    
    st.divider()
    
    # ==================== 主要指标卡 ====================
    # 计算指标
    total_comments_all = len(df)
    risk_comments = df['is_risk'].sum()
    negative_count = len(df[df['sentiment'] == '负向'])
    negative_ratio = (negative_count / total_comments_all * 100) if total_comments_all > 0 else 0
    
    # Top 品牌
    brand_counts = df['brand'].value_counts()
    top_brand = brand_counts.index[0] if len(brand_counts) > 0 else '无数据'
    top_brand_count = brand_counts.iloc[0] if len(brand_counts) > 0 else 0
    
    # 创建指标卡布局
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 总评论数",
            value=f"{total_comments_all:,}",
            delta=f"{len(df[df['date'] == df['date'].max()])} 今日"
        )
    
    with col2:
        st.metric(
            label="⚠️ 风险评论数",
            value=f"{risk_comments:,}",
            delta=f"{risk_comments/total_comments_all*100:.1f}%" if total_comments_all > 0 else "0%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="🔴 负向评论占比",
            value=f"{negative_ratio:.1f}%",
            delta=f"{negative_count} 条",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="🏆 评论量 Top 品牌",
            value=top_brand,
            delta=f"{top_brand_count:,} 条"
        )

def render_sentiment_charts(df):
    """渲染情感向分析图表"""
    st.subheader("📊 情感向分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 情感向饼图
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        
        # 自定义排序
        sentiment_order = ['正向', '中性', '负向']
        sentiment_counts['sentiment'] = pd.Categorical(
            sentiment_counts['sentiment'], 
            categories=sentiment_order, 
            ordered=True
        )
        sentiment_counts = sentiment_counts.sort_values('sentiment')
        
        fig_pie = px.pie(
            sentiment_counts,
            values='count',
            names='sentiment',
            title='情感向分布',
            color='sentiment',
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 品牌情感向堆叠柱状图
        brand_sentiment = df.groupby(['brand', 'sentiment']).size().reset_index(name='count')
        brand_sentiment['sentiment'] = pd.Categorical(
            brand_sentiment['sentiment'],
            categories=['正向', '中性', '负向'],
            ordered=True
        )
        
        fig_bar = px.bar(
            brand_sentiment,
            x='brand',
            y='count',
            color='sentiment',
            title='各品牌情感向分布',
            color_discrete_map=SENTIMENT_COLORS,
            barmode='stack'
        )
        fig_bar.update_layout(height=400, xaxis_title='品牌', yaxis_title='评论数')
        st.plotly_chart(fig_bar, use_container_width=True)

def render_trend_charts(df):
    """渲染趋势图表"""
    st.subheader("📈 舆论趋势分析")
    
    # 时间粒度选择
    time_granularity = st.radio(
        "时间粒度",
        options=['日', '周', '月'],
        horizontal=True,
        key='time_granularity'
    )
    
    if time_granularity == '日':
        df['time_key'] = df['date'].astype(str)
    elif time_granularity == '周':
        df['time_key'] = df['week']
    else:
        df['time_key'] = df['year_month']
    
    # 评论数量趋势
    trend_data = df.groupby('time_key').agg({
        'comment': 'count',
        'is_risk': 'sum'
    }).reset_index()
    trend_data.columns = ['time', 'total', 'risk']
    
    # 情感向趋势
    sentiment_trend = df.groupby(['time_key', 'sentiment']).size().unstack(fill_value=0).reset_index()
    
    fig_trend = go.Figure()
    
    # 总评论数
    fig_trend.add_trace(go.Scatter(
        x=trend_data['time'],
        y=trend_data['total'],
        mode='lines+markers',
        name='总评论数',
        line=dict(color=COLOR_SCHEME['primary'], width=2),
        marker=dict(size=6)
    ))
    
    # 风险评论数
    fig_trend.add_trace(go.Scatter(
        x=trend_data['time'],
        y=trend_data['risk'],
        mode='lines+markers',
        name='风险评论数',
        line=dict(color=COLOR_SCHEME['negative'], width=2),
        marker=dict(size=6)
    ))
    
    fig_trend.update_layout(
        title=f'评论数量趋势（按{time_granularity}）',
        xaxis_title='时间',
        yaxis_title='评论数',
        height=400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # 情感向趋势
    if len(sentiment_trend) > 0:
        fig_sentiment_trend = go.Figure()
        
        for sentiment in ['正向', '中性', '负向']:
            if sentiment in sentiment_trend.columns:
                fig_sentiment_trend.add_trace(go.Scatter(
                    x=sentiment_trend['time_key'],
                    y=sentiment_trend[sentiment],
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig_sentiment_trend.update_layout(
            title=f'情感向趋势（按{time_granularity}）',
            xaxis_title='时间',
            yaxis_title='评论数',
            height=400,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig_sentiment_trend, use_container_width=True)

def render_risk_type_charts(df):
    """渲染风险类型图表"""
    st.subheader("📋 风险类型分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 风险类型饼图
        type_counts = df['risk_type'].value_counts().reset_index()
        type_counts.columns = ['risk_type', 'count']
        
        fig_type_pie = px.pie(
            type_counts,
            values='count',
            names='risk_type',
            title='风险类型分布',
            color='risk_type',
            color_discrete_map=RISK_TYPE_COLORS
        )
        fig_type_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_type_pie.update_layout(height=400)
        st.plotly_chart(fig_type_pie, use_container_width=True)
    
    with col2:
        # 品牌风险类型堆叠柱状图
        brand_type = df.groupby(['brand', 'risk_type']).size().reset_index(name='count')
        
        fig_type_bar = px.bar(
            brand_type,
            x='brand',
            y='count',
            color='risk_type',
            title='各品牌风险类型构成',
            color_discrete_map=RISK_TYPE_COLORS,
            barmode='stack'
        )
        fig_type_bar.update_layout(height=400, xaxis_title='品牌', yaxis_title='评论数')
        st.plotly_chart(fig_type_bar, use_container_width=True)

def render_brand_distribution(df):
    """渲染品牌分布"""
    st.subheader("🏢 品牌评论分布")
    
    brand_counts = df['brand'].value_counts().reset_index()
    brand_counts.columns = ['brand', 'count']
    
    # 计算占比
    brand_counts['percentage'] = (brand_counts['count'] / brand_counts['count'].sum() * 100).round(1)
    
    fig_brand = px.bar(
        brand_counts,
        x='brand',
        y='count',
        title='各品牌评论数量',
        color='count',
        color_continuous_scale='Blues',
        text=brand_counts['percentage'].astype(str) + '%'
    )
    fig_brand.update_layout(height=400, xaxis_title='品牌', yaxis_title='评论数')
    fig_brand.update_traces(textposition='outside')
    st.plotly_chart(fig_brand, use_container_width=True)

def render_region_distribution(df):
    """渲染地区分布"""
    st.subheader("🗺️ 地区分布")
    
    region_counts = df['region'].value_counts().head(15).reset_index()
    region_counts.columns = ['region', 'count']
    
    fig_region = px.bar(
        region_counts,
        x='region',
        y='count',
        title='Top 15 地区评论数量',
        color='count',
        color_continuous_scale='Reds',
        orientation='v'
    )
    fig_region.update_layout(height=400, xaxis_title='地区', yaxis_title='评论数')
    st.plotly_chart(fig_region, use_container_width=True)

def render_product_analysis(df):
    """渲染产品分析"""
    st.subheader("🥤 产品分析")
    
    # Top 10 产品
    product_counts = df['product'].value_counts().head(10).reset_index()
    product_counts.columns = ['product', 'count']
    
    fig_product = px.bar(
        product_counts,
        x='count',
        y='product',
        title='Top 10 热门产品',
        color='count',
        color_continuous_scale='Greens',
        orientation='h'
    )
    fig_product.update_layout(height=400, yaxis_title='产品', xaxis_title='评论数')
    st.plotly_chart(fig_product, use_container_width=True)

# ==================== 词云模块 ====================

def render_wordcloud(df):
    """渲染词云"""
    st.subheader("☁️ 关键词分析")
    
    # 分情感向分析
    tab1, tab2, tab3 = st.tabs(["全部评论", "负向评论", "正向评论"])
    
    with tab1:
        generate_wordcloud_for_dataframe(df, "全部评论")
    
    with tab2:
        negative_df = df[df['sentiment'] == '负向']
        if len(negative_df) > 0:
            generate_wordcloud_for_dataframe(negative_df, "负向评论")
        else:
            st.info("暂无负向评论数据")
    
    with tab3:
        positive_df = df[df['sentiment'] == '正向']
        if len(positive_df) > 0:
            generate_wordcloud_for_dataframe(positive_df, "正向评论")
        else:
            st.info("暂无正向评论数据")

def generate_wordcloud_for_dataframe(df, title):
    """为指定数据生成词云"""
    if len(df) == 0:
        st.info("暂无数据")
        return
    
    # 合并所有评论
    all_text = ' '.join(df['comment'].astype(str))
    
    # 添加自定义词典
    custom_words = ['瑞幸', '星巴克', '喜茶', '奈雪', '蜜雪冰城', '茶百道', '古茗', 
                    '霸王茶姬', '沪上阿姨', '拿铁', '美式', '奶茶', '果茶']
    for word in custom_words:
        jieba.add_word(word)
    
    # 停用词
    stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', 
                     '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                     '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '又',
                     '与', '及', '等', '或', '但', '而', '因为', '所以', '如果',
                     '可以', '还是', '还是', '这个', '那个', '什么', '有点', '感觉',
                     '就是', '还是', '还是', '真的', '非常', '比较', '还是'])
    
    # 分词
    words = jieba.lcut(all_text)
    words = [w.strip() for w in words if len(w.strip()) > 1 and w.strip() not in stopwords]
    
    if len(words) == 0:
        st.info("无法提取有效关键词")
        return
    
    # 统计词频
    word_freq = pd.Series(words).value_counts().head(30)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 生成词云
        wordcloud_text = ' '.join(words)
        
        # 根据标题选择配色
        if '负向' in title:
            colormap = 'Reds'
        elif '正向' in title:
            colormap = 'Greens'
        else:
            colormap = 'Blues'
        
        # 字体路径：优先使用项目内的字体（兼容 GitHub 部署）
        import os
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        font_paths = [
            os.path.join(_script_dir, 'NotoSansSC-Regular.otf'),  # 项目内字体（跨平台）
            r'C:\Windows\Fonts\msyh.ttc',        # Windows 微软雅黑
            r'C:\Windows\Fonts\simhei.ttf',       # Windows 黑体
            r'C:\Windows\Fonts\simsun.ttc',       # Windows 宋体
            '/Library/Fonts/Arial Unicode.ttf',   # macOS
            '/System/Library/Fonts/PingFang.ttc', # macOS
        ]
        font_path = None
        for fp in font_paths:
            if os.path.exists(fp):
                font_path = fp
                break
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path=font_path,
            max_words=100,
            relative_scaling=0.5,
            colormap=colormap
        ).generate(wordcloud_text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        st.pyplot(fig)
    
    with col2:
        # Top 关键词表格
        st.markdown(f"**{title} - Top 30 关键词**")
        keywords_df = word_freq.reset_index()
        keywords_df.columns = ['关键词', '频次']
        st.dataframe(keywords_df, use_container_width=True, hide_index=True)

# ==================== 数据导出 ====================

def render_data_export(df):
    """渲染数据导出功能"""
    st.subheader("💾 数据导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV 导出
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载 CSV",
            data=csv,
            file_name=f"舆情数据_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    
    with col2:
        # Excel 导出
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='舆情数据')
        excel_data = output.getvalue()
        
        st.download_button(
            label="📥 下载 Excel",
            data=excel_data,
            file_name=f"舆情数据_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# ==================== 原始数据展示 ====================

def render_raw_data(df):
    """渲染原始数据表格"""
    st.subheader("📋 原始数据")
    
    # 选择显示的列
    display_cols = ['brand', 'product', 'datetime', 'comment', 'sentiment', 'risk_level', 'risk_type', 'region']
    available_cols = [col for col in display_cols if col in df.columns]
    
    # 分页显示
    page_size = st.selectbox("每页显示", options=[10, 20, 50, 100], index=1)
    
    # 显示数据
    st.dataframe(
        df[available_cols].head(page_size),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"共 {len(df)} 条数据")

# ==================== 主程序 ====================

def main():
    """主函数"""
    set_page_style()
    
    # 侧边栏 - 数据上传
    st.sidebar.header("📁 数据上传")
    
    # 默认数据文件路径（与脚本同目录）
    import os
    default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '产品评价_茶饮舆论.xlsx')
    
    uploaded_file = st.sidebar.file_uploader(
        "上传数据文件 (CSV/Excel)",
        type=['csv', 'xlsx', 'xls']
    )
    
    # 加载数据
    if uploaded_file is not None:
        # 保存上传的文件
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        df = load_and_process_data(tmp_path)
        st.sidebar.success("✅ 已加载上传的数据")
    else:
        # 尝试加载默认文件
        import os
        if os.path.exists(default_file):
            df = load_and_process_data(default_file)
            st.sidebar.success("✅ 已加载默认数据文件")
        else:
            st.sidebar.error("❌ 未找到数据文件，请上传")
            df = None
    
    if df is None:
        st.error("数据加载失败，请检查文件格式")
        return
    
    # 显示数据概览
    with st.sidebar.expander("👀 数据概览"):
        st.write(f"数据行数: {len(df):,}")
        st.write(f"时间范围: {df['date'].min()} ~ {df['date'].max()}")
        st.write(f"品牌数: {df['brand'].nunique()}")
        st.write(f"产品数: {df['product'].nunique()}")
        
        # 情感向分布
        st.write("情感向分布:")
        st.write(df['sentiment'].value_counts())
    
    # 渲染筛选器
    filtered_df = render_filters(df)
    
    # 筛选后无数据，显示空状态提示
    if len(filtered_df) == 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align: center; padding: 4rem 0;'>"
            "📭<h2 style='color: #999; margin: 1rem 0;'>当前筛选条件下无数据</h2>"
            "<p style='color: #aaa;'>请尝试调整侧边栏的时间范围、品牌、情感向等筛选条件</p>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(
            f"<div style='text-align: center; color: #666;'>"
            f"☕ 茶饮/咖啡品牌舆情风险分析 Dashboard | "
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"</div>",
            unsafe_allow_html=True
        )
        return
    
    # 主内容区
    render_overview_metrics(filtered_df)
    
    st.divider()
    
    # 1. 产品正负面占比
    render_sentiment_charts(filtered_df)
    
    st.divider()
    
    # 2. 同行类似品表现比对
    render_brand_distribution(filtered_df)
    
    st.divider()
    
    # 3. 舆论声量趋势变化
    render_trend_charts(filtered_df)
    
    st.divider()
    
    # 4. 区域分布
    render_region_distribution(filtered_df)
    
    st.divider()
    
    # 5. 单品热度排行榜
    render_product_analysis(filtered_df)
    
    st.divider()
    
    # 6. 问题类型分布
    render_risk_type_charts(filtered_df)
    
    st.divider()
    
    # 7. 关键词分析
    render_wordcloud(filtered_df)
    
    st.divider()
    
    # 原始数据
    render_raw_data(filtered_df)
    
    st.divider()
    
    # 数据导出
    render_data_export(filtered_df)
    
    # 页脚
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"☕ 茶饮/咖啡品牌舆情风险分析 Dashboard | "
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
