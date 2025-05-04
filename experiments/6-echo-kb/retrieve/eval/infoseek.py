def calculate_dataset_recall_at_k(gts: list, retrieved: list, ks: list):
    """
    Calculate micro and macro recall@k for an entire dataset.
    
    Args:
        gts: list of ground truth relevant results (one per query)
        retrieved: list of retrieved results lists (one list per query)
        ks: list of k values to calculate recall for
    
    Returns:
        dict: contains both micro and macro recall@k results
    """
    assert len(gts) == len(retrieved), "GT and retrieved must have same length"
    
    # Calculate individual recalls for macro-average
    macro_recalls = {k: [] for k in ks}
    
    for gt, retrieved_list in zip(gts, retrieved):
        query_recalls = calculate_single_recall_at_k(gt, retrieved_list, ks)
        for k in ks:
            macro_recalls[k].append(query_recalls[k])
    
    # Calculate macro-average
    macro_results = {}
    for k in ks:
        macro_results[k] = sum(macro_recalls[k]) / len(macro_recalls[k])
    
    # Calculate micro-average
    micro_results = calculate_micro_recall_at_k(gts, retrieved, ks)
    
    return {
        'macro': macro_results,
        'micro': micro_results
    }

def calculate_single_recall_at_k(gt: str, retrieved: list, ks: list):
    """Calculate recall@k for a single query."""
    recalls = {}
    for k in ks:
        retrieved_at_k = retrieved[:k]
        recalls[k] = 1 if gt in retrieved_at_k else 0
    return recalls

def calculate_micro_recall_at_k(gts: list, retrieved: list, ks: list):
    """Calculate micro-average recall@k."""
    # Combine all gt and retrieved results with query indices
    combined_gt = []
    combined_retrieved = []
    
    for i, (gt, retrieved_list) in enumerate(zip(gts, retrieved)):
        # Add query index to each item to preserve uniqueness
        combined_gt.append(f"{i}_{gt}")
        combined_retrieved.extend([f"{i}_{item}" for item in retrieved_list])
    
    # Calculate recall@k for the combined dataset
    micro_results = {}
    for k in ks:
        # For micro-average, k needs to be applied to each query separately
        relevant_count = 0
        total_queries = len(gts)
        
        for i, (gt, retrieved_list) in enumerate(zip(gts, retrieved)):
            if len(retrieved_list) >= k and gt in retrieved_list[:k]:
                relevant_count += 1
            elif len(retrieved_list) < k and gt in retrieved_list:
                relevant_count += 1
        
        micro_results[k] = relevant_count / total_queries
    
    return micro_results

import pickle
import sys
sys.path.append("/data/track/mm-RewardRAG")
test_file = "/data/dataset/mm_RewardRAG/InfoSeek/valid.json"

import pickle

id_to_title_map = pickle.load(open("/data/track/mm-RewardRAG/experiments/6-echo-kb/retrieve/eval/infoseek_id_to_title.pkl", "rb"))

import json
test_data = json.load(open(test_file, "r"))
gt_list = [id_to_title_map[item["entity_id"]] for item in test_data]

retrieved_file = "/data/track/mm-RewardRAG/experiments/6-echo-kb/retrieve/results/infoseek/retrieved.pkl"
retrieved_data = pickle.load(open(retrieved_file, "rb"))
retrieved_list = [[item.title for item in single_question] for single_question in retrieved_data]

k = [1, 3, 5, 10, 20, 50, 100]
results = calculate_dataset_recall_at_k(gt_list, retrieved_list, k)
print("Results for recall@k:")
for recall_type, recalls in results.items():
    print(f"{recall_type} recall@k:")
    for k_val, recall in recalls.items():
        print(f"Recall@{k_val}: {recall:.4f}")

