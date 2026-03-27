"""
TF-IDF 情感分类器预测脚本
用于批量预测新的茶饮评论情感向
"""

import os
import sys
import re
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import jieba

# 默认模型路径（相对于脚本位置）
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# ID到标签的映射
ID2LABEL = {0: '负向', 1: '中性', 2: '正向'}


class SentimentPredictor:
    """情感分类预测器 (TF-IDF版)"""
    
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        """加载模型"""
        print(f"加载模型: {model_dir}")
        
        # 加载配置
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载vectorizer
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # 加载模型
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"模型类型: {self.config['model_type']}")
        print(f"特征维度: {self.config['feature_dim']}")
        print(f"训练样本: {self.config['n_samples']}")
        print(f"测试准确率: {self.config['accuracy']:.2%}")
        print("模型加载完成")
    
    def clean_text(self, text):
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[@#][^\s]+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text) > 0 else "无"
    
    def tokenize(self, text):
        """jieba分词"""
        return ' '.join(jieba.cut(text))
    
    def predict(self, texts, return_probs=False):
        """
        批量预测
        
        Args:
            texts: 文本列表或单个文本
            return_probs: 是否返回概率
            
        Returns:
            如果 return_probs=False: ['正向', '负向', ...]
            如果 return_probs=True: (['正向', ...], [[0.1, 0.2, 0.7], ...])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 清洗并分词
        clean_texts = [self.clean_text(t) for t in texts]
        tokens = [self.tokenize(t) for t in clean_texts]
        
        # 特征提取
        X = self.vectorizer.transform(tokens)
        
        # 预测
        preds = self.model.predict(X)
        labels = [ID2LABEL[p] for p in preds]
        
        if return_probs:
            probs = self.model.predict_proba(X)
            return labels, probs.tolist()
        return labels
    
    def predict_excel(self, excel_path, text_column='评论内容', output_path=None):
        """
        预测整个Excel文件
        
        Args:
            excel_path: Excel文件路径
            text_column: 评论内容所在列名
            output_path: 输出路径（默认覆盖原文件）
        """
        print(f"\n加载Excel: {excel_path}")
        df = pd.read_excel(excel_path, sheet_name='茶饮舆论')
        
        print(f"总数据: {len(df)} 条")
        
        # 只预测空情感向的行
        mask = df['情感向'].isna()
        to_predict = df[mask]
        
        if len(to_predict) == 0:
            print("没有需要预测的空白数据")
            return df
        
        print(f"待预测: {len(to_predict)} 条")
        
        # 分批预测
        batch_size = 1000
        texts = to_predict[text_column].fillna('').tolist()
        all_preds = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            preds = self.predict(batch)
            all_preds.extend(preds)
            if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(texts):
                print(f"  已处理: {min(i+batch_size, len(texts))}/{len(texts)}")
        
        # 填充结果
        df.loc[mask, '情感向'] = all_preds
        
        # 保存
        if output_path is None:
            output_path = excel_path
        
        # 保留其他sheet
        all_sheets = pd.read_excel(excel_path, sheet_name=None)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='茶饮舆论', index=False)
            for sname, sdf in all_sheets.items():
                if sname != '茶饮舆论':
                    sdf.to_excel(writer, sheet_name=sname, index=False)
        
        print(f"已保存: {output_path}")
        
        # 打印分布
        print("\n情感向分布:")
        print(df['情感向'].value_counts())
        
        return df


def demo():
    """演示预测"""
    predictor = SentimentPredictor()
    
    test_texts = [
        "这个奶茶真的太好喝了，强烈推荐！",
        "一般般，没什么特别的",
        "太难喝了，完全踩雷，不会再买",
        "口感不错，就是有点贵",
        "喝了拉肚子，不推荐",
        "yyds，一口上瘾",
        "性价比太低，不值这个价",
        "还行吧，无功无过"
    ]
    
    print("\n" + "=" * 60)
    print("预测示例")
    print("=" * 60)
    
    labels, probs = predictor.predict(test_texts, return_probs=True)
    
    for text, label, prob in zip(test_texts, labels, probs):
        prob_str = f"[负:{prob[0]:.2f} 中:{prob[1]:.2f} 正:{prob[2]:.2f}]"
        print(f"\n文本: {text}")
        print(f"预测: {label} {prob_str}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 命令行模式: python predict_tfidf_sentiment.py <excel_path>
        excel_path = sys.argv[1]
        predictor = SentimentPredictor()
        predictor.predict_excel(excel_path)
    else:
        # 演示模式
        demo()
