# 茶饮/咖啡品牌评论分析 Dashboard

基于小红书评论数据的茶饮/咖啡品牌舆情风险分析看板，使用 Streamlit + Plotly 构建。

## 功能特性

- 📊 **情感分类**：通过 TF-IDF + 逻辑回归模型自动对评论内容进行正向/中性/负向三分类
- 🚨 **单品热度排行**：按负向占比对产品进行风险评分排名
- 📈 **趋势分析**：支持日/周/月粒度的评论声量与情感趋势
- 🗺️ **地区分布**：Top 15 评论来源地区
- ☁️ **词云分析**：全部/正向/负向评论的关键词词云
- 🔍 **级联筛选**：品牌 → 产品 → 时间范围 → 情感向 → 评论类型，逐级过滤

## 目录结构

```
dashboard_deploy/
├── sentiment_dashboard.py   # 主程序
├── predict_sentiment.py     # 情感分类预测脚本
├── model/
│   ├── model.pkl            # 逻辑回归分类器
│   ├── vectorizer.pkl       # TF-IDF 特征提取器
│   └── config.json          # 模型配置
├── NotoSansSC-VF.ttf        # 中文字体（词云使用）
├── requirements.txt
└── .gitignore
```

> **数据文件**：将 `产品评价_茶饮舆论.xlsx` 放在项目根目录，启动后自动加载；也可通过侧边栏手动上传。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动看板
streamlit run sentiment_dashboard.py
```

## 情感分类模型

| 项目 | 值 |
|------|-----|
| 模型类型 | TF-IDF + LogisticRegression |
| 训练样本 | 30,129 条小红书茶饮评论 |
| 测试准确率 | 70.15% |
| 分类标签 | 正向 / 中性 / 负向 |
