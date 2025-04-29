# 这里现在是假设得到了一个list，每个list是一个dict，就是{"cot","queries","key_words_string"},还有一个gt entity id已知；

import numpy as np
import json

def extract_json_fields(response_text):
    try:
        # Try to extract JSON from code blocks
        if "```json" in response_text and "```" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            # Use regex to find JSON object
            import re
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None, None  # No JSON found
        
        # Parse JSON and extract fields
        data = json.loads(json_str)
        queries = data.get("queries", [])
        key_words_string = data.get("key_words_string", "")
        
        return queries, key_words_string
    
    except json.JSONDecodeError:
        return None, None  # Invalid JSON
    except Exception as e:
        return None, None  # Any other error

def score_ranking(gt_ids, mixed_list, method='flexible', position_weight_factor=1.0, relevance_scores=None):
    """
    计算混合列表中ground truth IDs的排名得分。
    
    Args:
        gt_ids (list): ground truth ID列表
        mixed_list (list): 包含ground truth和hard negatives的混合列表
        method (str): 评分方法，可选 'flexible'(默认),'map'或'ndcg'
        position_weight_factor (float): 位置权重因子，控制位置重要性（用于flexible方法）
                                       较大的值会更重视靠前的位置
                                       较小的值会更均衡地考虑所有召回的项
        relevance_scores (dict, optional): 每个gt_id的相关性分数字典，用于NDCG计算
                                          如果为None，则所有相关项的相关性均为1
    
    Returns:
        float: 反映ground truth IDs排名质量的得分
    """
    if len(mixed_list) == 0:
        return 0.0
    
    # 如果relevance_scores未提供，则默认所有gt_ids的相关性为1
    if relevance_scores is None:
        relevance_scores = {gt_id: 1.0 for gt_id in gt_ids}
    
    if method == 'map':
        # 计算MAP (Mean Average Precision)
        precisions = []
        relevant_count = 0
        
        for i, item_id in enumerate(mixed_list):
            if item_id in gt_ids:
                relevant_count += 1
                # 计算当前位置的精确度 (precision)
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)
        
        # 计算平均精确度 (AP)
        if len(precisions) > 0:
            return sum(precisions) / len(gt_ids)  # 除以相关文档总数
        else:
            return 0.0
            
    elif method == 'ndcg':
        # 计算NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        
        # 计算DCG
        for i, item_id in enumerate(mixed_list):
            if item_id in gt_ids:
                # 获取相关性分数
                rel = relevance_scores.get(item_id, 1.0)
                
                # 位置从0开始，所以位置i对应的是第i+1个元素
                # DCG计算公式: DCG = sum(rel_i / log2(i+2))，从i=0开始
                # 注意: log2(i+2)是为了使分母从log2(2)开始，即从1开始
                if i == 0:
                    dcg += rel  # 第一个位置的处理
                else:
                    dcg += rel / np.log2(i + 2)
        
        # 构建理想排序的列表
        relevant_items = [item_id for item_id in gt_ids if item_id in relevance_scores]
        relevant_items.sort(key=lambda x: relevance_scores.get(x, 1.0), reverse=True)
        
        # 计算理想DCG (IDCG)
        idcg = 0.0
        for i, item_id in enumerate(relevant_items[:len(mixed_list)]):  # 截断到mixed_list长度
            rel = relevance_scores.get(item_id, 1.0)
            if i == 0:
                idcg += rel
            else:
                idcg += rel / np.log2(i + 2)
        
        # 计算NDCG
        if idcg > 0:
            return dcg / idcg
        else:
            return 0.0
            
    else:  # method == 'flexible'
        # 自定义评分函数，结合位置权重
        score = 0.0
        
        for i, item_id in enumerate(mixed_list):
            if item_id in gt_ids:
                # 位置权重：位置越靠前，权重越大
                position_weight = 1.0 / ((i + 1) ** position_weight_factor)
                # 如果提供了相关性分数，则使用它
                relevance = relevance_scores.get(item_id, 1.0)
                score += position_weight * relevance
        
        return score

# 示例使用
if __name__ == "__main__":
    gt_ids = [1, 2, 3]
    
    # 案例1：只有一个gt，但位置靠前
    mixed_list1 = [1, 10, 11, 12, 13]
    
    # 案例2：有两个gt，但位置靠后
    mixed_list2 = [10, 11, 1, 2, 13]
    
    # 定义每个gt_id的相关性分数（用于NDCG）
    relevance_scores = {1: 3.0, 2: 2.0, 3: 1.0}  # ID 1最相关，ID 3最不相关
    
    # 使用不同方法和参数评分
    print("使用灵活方法（默认位置权重）:")
    print(f"案例1得分: {score_ranking(gt_ids, mixed_list1, 'flexible')}")
    print(f"案例2得分: {score_ranking(gt_ids, mixed_list2, 'flexible')}")
    
    print("\n更重视位置:")
    print(f"案例1得分: {score_ranking(gt_ids, mixed_list1, 'flexible', position_weight_factor=2.0)}")
    print(f"案例2得分: {score_ranking(gt_ids, mixed_list2, 'flexible', position_weight_factor=2.0)}")
    
    print("\n使用MAP方法:")
    print(f"案例1得分: {score_ranking(gt_ids, mixed_list1, 'map')}")
    print(f"案例2得分: {score_ranking(gt_ids, mixed_list2, 'map')}")
    
    print("\n使用NDCG方法（带相关性分数）:")
    print(f"案例1得分: {score_ranking(gt_ids, mixed_list1, 'ndcg', relevance_scores=relevance_scores)}")
    print(f"案例2得分: {score_ranking(gt_ids, mixed_list2, 'ndcg', relevance_scores=relevance_scores)}")