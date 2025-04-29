import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from sklearn.metrics import ndcg_score, average_precision_score, roc_auc_score

def evaluate_reranker_dataset(
    all_ground_truth_ids: List[List[str]],
    all_reranked_ids: List[List[str]],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    评估整个数据集上reranker性能的函数
    
    参数:
        all_ground_truth_ids: 包含多个查询的ground truth ID列表的列表
        all_reranked_ids: 包含多个查询的rerank后ID列表的列表
        metrics: 要计算的指标列表，默认为None时计算所有指标
        
    返回:
        包含各项指标平均值的字典
    """
    if metrics is None:
        metrics = ["ndcg", "ndcg@3", "ndcg@5", "ndcg@10", 
                  "map", "mrr", "p@1", "p@3", "p@5", "p@10", "recall@k"]
    
    # 确保两个列表长度相同
    assert len(all_ground_truth_ids) == len(all_reranked_ids), "输入列表长度不匹配"
    
    # 存储所有查询的结果
    all_results = []
    
    # 对每个查询进行评估
    for gt_ids, reranked_ids in zip(all_ground_truth_ids, all_reranked_ids):
        # 调用单条数据评估函数
        query_results = evaluate_reranker(gt_ids, reranked_ids, metrics)
        all_results.append(query_results)
    
    # 计算所有指标的平均值
    avg_results = {}
    
    # 处理常规指标
    for metric in metrics:
        if metric != "recall@k":
            # 收集所有查询在此指标上的值
            metric_values = []
            for result in all_results:
                if metric in result:
                    metric_values.append(result[metric])
            
            # 计算平均值
            if metric_values:
                avg_results[metric] = float(np.mean(metric_values))
            else:
                avg_results[metric] = 0.0
    
    # 特别处理recall@k
    if "recall@k" in metrics:
        avg_results["recall@k"] = {}
        for k in [1, 3, 5, 10]:
            recall_key = f"recall@{k}"
            recall_values = []
            
            for result in all_results:
                if "recall@k" in result and recall_key in result["recall@k"]:
                    recall_values.append(result["recall@k"][recall_key])
            
            if recall_values:
                avg_results["recall@k"][recall_key] = float(np.mean(recall_values))
            else:
                avg_results["recall@k"][recall_key] = 0.0
    
    return avg_results

def evaluate_reranker(
    ground_truth_ids: List[str],
    reranked_ids: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    评估单个查询的reranker性能函数
    
    参数:
        ground_truth_ids: 相关文档ID列表
        reranked_ids: rerank后的ID列表（包含ground truth ids和hard negative ids）
        metrics: 要计算的指标列表，默认为None时计算所有指标
        
    返回:
        包含各项指标的字典
    """
    if metrics is None:
        metrics = ["ndcg", "ndcg@3", "ndcg@5", "ndcg@10", 
                  "map", "mrr", "p@1", "p@3", "p@5", "p@10", "recall@k"]
    
    results = {}
    
    # 创建相关性标签列表 (1表示在ground truth中，0表示不在)
    relevance = np.zeros(len(reranked_ids))
    for i, doc_id in enumerate(reranked_ids):
        if doc_id in ground_truth_ids:
            relevance[i] = 1
    
    # 计算NDCG (Normalized Discounted Cumulative Gain)
    if "ndcg" in metrics:
        if np.sum(relevance) > 0:  # 确保至少有一个相关文档
            # 使用全长度计算NDCG
            results["ndcg"] = _calculate_ndcg(relevance)
    
    # 计算不同k值的NDCG
    for metric in metrics:
        if metric.startswith("ndcg@"):
            k = int(metric.split("@")[1])
            if k <= len(relevance):
                results[metric] = _calculate_ndcg(relevance[:k])
    
    # 计算MAP (Mean Average Precision)
    if "map" in metrics and np.sum(relevance) > 0:
        results["map"] = _calculate_map(relevance)
    
    # 计算MRR (Mean Reciprocal Rank)
    if "mrr" in metrics:
        results["mrr"] = _calculate_mrr(relevance)
    
    # 计算P@k (Precision at k)
    for metric in metrics:
        if metric.startswith("p@"):
            k = int(metric.split("@")[1])
            if k <= len(relevance):
                results[metric] = np.mean(relevance[:k])
    
    # 计算Recall@k
    if "recall@k" in metrics:
        total_relevant = len(ground_truth_ids)
        results["recall@k"] = {}
        for k in [1, 3, 5, 10]:
            if k <= len(relevance):
                relevant_at_k = np.sum(relevance[:k])
                results["recall@k"][f"recall@{k}"] = float(relevant_at_k / total_relevant) if total_relevant > 0 else 0.0
    
    return results

def _calculate_ndcg(relevance: np.ndarray) -> float:
    """计算NDCG"""
    if np.sum(relevance) == 0:
        return 0.0
    
    # 创建理想排序（所有相关文档排在前面）
    ideal_relevance = np.sort(relevance)[::-1]
    
    # 使用sklearn的ndcg_score计算NDCG
    return float(ndcg_score(np.array([ideal_relevance]), np.array([relevance])))

def _calculate_map(relevance: np.ndarray) -> float:
    """计算MAP"""
    if np.sum(relevance) == 0:
        return 0.0
    
    # 计算每个位置的精度
    precisions = np.cumsum(relevance) / np.arange(1, len(relevance) + 1)
    
    # 只考虑相关文档位置的精度
    ap = np.sum(precisions * relevance) / np.sum(relevance)
    return float(ap)

def _calculate_mrr(relevance: np.ndarray) -> float:
    """计算MRR"""
    # 找到第一个相关文档的位置
    relevant_indices = np.where(relevance == 1)[0]
    if len(relevant_indices) == 0:
        return 0.0
    
    # 计算倒数排名
    first_relevant_idx = relevant_indices[0]
    return float(1.0 / (first_relevant_idx + 1))  # +1因为索引从0开始

# 使用示例
if __name__ == "__main__":
    # 示例数据集
    all_ground_truth_ids = [
        ["doc1", "doc2", "doc3"],
        ["doc7", "doc8"],
        ["doc12", "doc13", "doc14", "doc15"]
    ]
    
    all_reranked_ids = [
        ["doc2", "doc5", "doc1", "doc4", "doc3", "doc6"],
        ["doc9", "doc7", "doc10", "doc8", "doc11"],
        ["doc12", "doc16", "doc14", "doc17", "doc13", "doc15"]
    ]
    
    # 评估整个数据集的结果
    dataset_results = evaluate_reranker_dataset(all_ground_truth_ids, all_reranked_ids)
    print("数据集评估结果:")
    print(dataset_results)