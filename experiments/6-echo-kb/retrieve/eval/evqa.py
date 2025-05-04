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
    
    # Validate k values
    for k in ks:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
    
    # Calculate individual recalls for macro-average
    macro_recalls = {k: [] for k in ks}
    
    for gt, retrieved_list in zip(gts, retrieved):
        query_recalls = calculate_single_recall_at_k(gt, retrieved_list, ks)
        for k in ks:
            macro_recalls[k].append(query_recalls[k])
    
    # Calculate macro-average
    macro_results = {}
    for k in ks:
        macro_results[k] = sum(macro_recalls[k]) / len(macro_recalls[k]) if macro_recalls[k] else 0
    
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
        if not retrieved:  # Handle empty retrieved list
            recalls[k] = 0
        else:
            retrieved_at_k = retrieved[:k]
            recalls[k] = 1 if gt in retrieved_at_k else 0
    return recalls

def calculate_micro_recall_at_k(gts: list, retrieved: list, ks: list):
    """Calculate micro-average recall@k."""
    micro_results = {}
    
    for k in ks:
        relevant_at_k = 0
        total_gt_count = len(gts)
        
        for gt, retrieved_list in zip(gts, retrieved):
            # Check if gt is in top k retrieved results
            if retrieved_list and gt in retrieved_list[:k]:
                relevant_at_k += 1
        
        # Micro-average = total relevant found / total relevant possible
        micro_results[k] = relevant_at_k / total_gt_count if total_gt_count > 0 else 0
    
    return micro_results

# Your main code
import pickle
import sys
sys.path.append("/data/track/mm-RewardRAG")
test_file = "/data/track/mm-RewardRAG/eval_datasets/encyclopedia/test.json"

import json
test_data = json.load(open(test_file, "r"))
gt_list = [item["wikipedia_title"] for item in test_data]

retrieved_file = "/data/track/mm-RewardRAG/experiments/6-echo-kb/retrieve/results/evqa/retrieved.pkl"
retrieved_data = pickle.load(open(retrieved_file, "rb"))
retrieved_list = [[item.title for item in single_question] for single_question in retrieved_data]

k = [1, 3, 5, 10, 20, 50, 100]
results = calculate_dataset_recall_at_k(gt_list, retrieved_list, k)

print("Results for recall@k:")
print("=" * 50)
for recall_type, recalls in results.items():
    print(f"\n{recall_type.upper()} recall@k:")
    for k_val, recall in recalls.items():
        print(f"  Recall@{k_val}: {recall:.4f}")
