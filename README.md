# ☕ 茶饮/咖啡品牌舆情风险分析 Dashboard

基于小红书评论数据，对茶饮/咖啡品牌进行舆情风险可视化分析的 Streamlit 网页应用。

## 功能特性

- 🚨 **产品风险等级排行榜** — 按负向评论占比自动排序
- 📊 **情感向分析** — 正向/中性/负向分布饼图 & 各品牌堆叠柱状图
- 📈 **舆论趋势分析** — 支持日/周/月粒度切换
- 🏢 **品牌评论分布** — 各品牌声量对比
- 🗺️ **地区分布** — Top 15 评论来源地区
- 🥤 **产品分析** — Top 10 热门产品
- 📋 **风险类型分析** — 口味/包装/口感/服务/价格/食安分类
- ☁️ **关键词词云** — 全部/负向/正向分Tab展示
- 💾 **数据导出** — 支持 CSV / Excel 下载

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将数据文件（Excel 格式）放到与 `sentiment_dashboard.py` 相同的目录，文件名为：

```
产品评价_茶饮舆论.xlsx
```

数据需包含以下列：

| 列名 | 说明 |
|------|------|
| 品牌 | 品牌名称 |
| 产品 | 产品名称 |
| 评论内容 | 用户评论文本 |
| 评论日期 | 评论时间 |
| IP 属地 | 评论来源地区 |
| 情感向 | 正向 / 中性 / 负向 |

> 也可以在运行后通过侧边栏上传自定义数据文件（CSV 或 Excel）。

### 3. 启动应用

```bash
streamlit run sentiment_dashboard.py
```

浏览器访问 `http://localhost:8501`

## 部署到 Streamlit Cloud（免费公网访问）

1. Fork 本仓库到你的 GitHub
2. 前往 [share.streamlit.io](https://share.streamlit.io) 登录
3. 选择仓库和 `sentiment_dashboard.py` 文件
4. 点击 Deploy，获取公网链接

> ⚠️ 云端部署时无本地数据文件，请通过侧边栏上传数据。

## 技术栈

- [Streamlit](https://streamlit.io/) — 网页框架
- [Plotly](https://plotly.com/) — 交互式图表
- [jieba](https://github.com/fxsjy/jieba) — 中文分词
- [WordCloud](https://github.com/amueller/word_cloud) — 词云生成
- Pandas / NumPy — 数据处理
