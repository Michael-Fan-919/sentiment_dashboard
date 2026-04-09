"""
情感预测模块 - Baseline 版本
直接使用 TfidfVectorizer + LinearSVC，无需分词
"""

import os
import json
import pickle
from pathlib import Path
from typing import Union, List

# 模型文件路径
MODEL_DIR = Path(__file__).parent / "model"

class SentimentPredictor:
    """情感分析预测器 - Baseline 版本"""
    
    def __init__(self, model_dir: Union[str, Path] = None):
        """
        初始化预测器
        
        Args:
            model_dir: 模型目录路径，默认使用当前目录下的 model 文件夹
        """
        if model_dir is None:
            model_dir = MODEL_DIR
        self.model_dir = Path(model_dir)
        
        # 加载配置
        config_path = self.model_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # 加载 vectorizer
        with open(self.model_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        
        # 加载模型
        with open(self.model_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        
        self.labels = self.config["labels"]
        print(f"[SentimentPredictor] 模型加载成功: {self.model_dir}")
    
    def predict(self, texts: Union[str, List[str]]) -> List[dict]:
        """
        预测情感
        
        Args:
            texts: 单条文本或文本列表
        
        Returns:
            预测结果列表，每个元素包含:
            - text: 原始文本
            - label: 预测标签 (负向/中性/正向)
            - confidence: 置信度 (基于决策函数距离)
        """
        # 统一转换为列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 清洗文本
        texts = [str(t).strip() for t in texts]
        
        # 直接 vectorizer.transform - 不做分词拼接
        X = self.vectorizer.transform(texts)
        
        # 预测标签
        predictions = self.model.predict(X)
        
        # 获取决策函数值作为置信度参考
        decision_values = self.model.decision_function(X)
        
        # 构建结果
        results = []
        
        # 获取模型类别顺序
        if hasattr(self.model, 'classes_'):
            model_classes = list(self.model.classes_)
        else:
            model_classes = self.labels
        
        for i, (text, label) in enumerate(zip(texts, predictions)):
            # 获取决策函数值
            if decision_values.ndim == 1:
                dv = decision_values
            else:
                dv = decision_values[i]
            
            # 使用 softmax 转换为概率
            import math
            max_dv = max(dv)
            exp_dv = [math.exp(v - max_dv) for v in dv]
            sum_exp = sum(exp_dv)
            probs = [e / sum_exp for e in exp_dv]
            
            # 按 model_classes 顺序构建概率字典
            prob_dict = {
                cls: round(prob, 4) 
                for cls, prob in zip(model_classes, probs)
            }
            
            confidence = prob_dict.get(label, 0.0)
            
            results.append({
                "text": text,
                "label": label,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict
            })
        
        return results
    
    def predict_batch(self, texts: List[str], batch_size: int = 1000) -> List[dict]:
        """
        批量预测（大数据集分批次处理）
        
        Args:
            texts: 文本列表
            batch_size: 每批处理的样本数
        
        Returns:
            预测结果列表
        """
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.predict(batch)
            all_results.extend(results)
        return all_results


# 全局预测器实例（延迟加载）
_predictor = None

def get_predictor() -> SentimentPredictor:
    """获取全局预测器实例"""
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor


def predict(texts: Union[str, List[str]]) -> List[dict]:
    """
    便捷预测函数
    
    Args:
        texts: 单条文本或文本列表
    
    Returns:
        预测结果列表
    
    Example:
        >>> predict("这个奶茶真好喝")
        [{'text': '这个奶茶真好喝', 'label': '正向', 'confidence': 0.9234, ...}]
        
        >>> predict(["太难喝了", "一般般", "超喜欢"])
        [{'text': '太难喝了', 'label': '负向', ...}, ...]
    """
    return get_predictor().predict(texts)


def predict_label(text: str) -> str:
    """
    只返回标签的便捷函数
    
    Args:
        text: 输入文本
    
    Returns:
        预测标签 (负向/中性/正向)
    """
    result = get_predictor().predict(text)
    return result[0]["label"]


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("情感预测测试")
    print("=" * 60)
    
    test_texts = [
        "这个奶茶真好喝",
        "太难喝了，浪费钱",
        "一般般吧",
        "超喜欢",
        "还行",
        "不推荐"
    ]
    
    results = predict(test_texts)
    
    print("\n预测结果:")
    for r in results:
        print(f"  文本: {r['text']}")
        print(f"  标签: {r['label']} (置信度: {r['confidence']})")
        print(f"  概率: {r['probabilities']}")
        print()
