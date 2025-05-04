def calculate_retrieval_metrics(
    retrieved_entities, ground_truth_entity, k_values=[1, 3, 5, 10, 20, 50, 100]
):
    """
    Calculate retrieval metrics (Precision@K, Recall@K, Hit@K) for retrieved results.

    Args:
        retrieved_entities: List of retrieved entity IDs
        ground_truth_entity: Ground truth entity ID
        k_values: List of K values to calculate metrics for

    Returns:
        Dictionary containing metrics for each K value
    """
    metrics = {}

    # Sort k_values to ensure we don't recalculate unnecessarily
    k_values = sorted(k_values)
    max_k = max(k_values)

    # Ensure we don't try to evaluate beyond the number of retrieved items
    max_k = min(max_k, len(retrieved_entities))

    # For each k value
    for k in k_values:
        if k > len(retrieved_entities):
            continue

        # Get top-k retrieved entities
        top_k = retrieved_entities[:k]

        # Calculate Hit@K (1 if ground truth is in top k results, 0 otherwise)
        hit_at_k = 1 if ground_truth_entity in top_k else 0

        # Calculate Precision@K (fraction of relevant items among retrieved items)
        precision_at_k = sum(1 for entity in top_k if entity == ground_truth_entity) / k

        # Calculate Recall@K (1 if ground truth is found, 0 otherwise since we only have one relevant item)
        recall_at_k = 1 if ground_truth_entity in top_k else 0

        metrics[k] = {
            f"hit@{k}": hit_at_k,
            f"precision@{k}": precision_at_k,
            f"recall@{k}": recall_at_k,
        }

    return metrics


def evaluate_batch(retrieved_results, ground_truth_pairs):
    """
    Evaluate retrieval metrics for a batch of queries.

    Args:
        retrieved_results: Dictionary mapping data_id to list of retrieved entity IDs
        ground_truth_pairs: Dictionary mapping data_id to ground truth entity ID

    Returns:
        Dictionary containing averaged metrics across all queries
    """
    all_metrics = {}

    for data_id, gt_entity in ground_truth_pairs.items():
        if data_id not in retrieved_results:
            continue

        retrieved = retrieved_results[data_id]
        metrics = calculate_retrieval_metrics(retrieved, gt_entity)

        for k, k_metrics in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(k_metrics)

    # Calculate averages
    averaged_metrics = {}
    for k, metrics_list in all_metrics.items():
        if not metrics_list:
            continue

        averaged_metrics[k] = {
            metric: sum(m[metric] for m in metrics_list) / len(metrics_list)
            for metric in metrics_list[0].keys()
        }

    return averaged_metrics


def main():
    import json
    import os
    import numpy as np
    from tqdm import tqdm

    valid_dataset = json.load(
        open("/data/dataset/mm_RewardRAG/InfoSeek/valid.json", "r")
    )
    retrieved_gt = [
        json.loads(line)
        for line in open(
            "/data/track/mm-RewardRAG/eval/infoseek/infoseek_val_withkb.jsonl"
        )
    ]
    ground_truth_pairs = {item["data_id"]: item["entity_id"] for item in retrieved_gt}
    # 这里不能直接使用valid_dataset里面的，好像是因为有些数据很脏，根本没有对应的entity_id，那这样子的话，是不是在eval question的时候也不能直接用valid这个数据集？

    retrieved_dir = (
        "/data/track/mm-RewardRAG/experiments/0-retrieval/results/infoseek/q2img"
    )
    retrieved_results_path = os.path.join(retrieved_dir, "retrieved_results.json")
    query_vectors_dir = os.path.join(retrieved_dir, "query_tensors")

    # Import necessary modules for retrieval process

    # Check if retrieved results already exist
    if os.path.exists(retrieved_results_path):
        print(f"Loading existing retrieval results from {retrieved_results_path}")
        retrieved_results = json.load(open(retrieved_results_path))

    else:
        import sys

        sys.path.append("/data/track/mm-RewardRAG")
        from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
        from core.indexing.faiss_indexer import FaissIndexer
        from PIL import Image

        embedder = JinaClipV2Embedder()
        indexer = FaissIndexer(mode="jina-clip-v2")
        indexer.load(
            "/data/track/mm-RewardRAG/indexs/jina-clip-v2/visual_index_gpu/infoseek"
        )

        print("Generating retrieval results...")
        bs = 512
        retrieved_results = []

        # Create query_tensors_dir if it doesn't exist
        os.makedirs(query_vectors_dir, exist_ok=True)

        for i in tqdm(range(0, len(valid_dataset), bs)):
            batch = valid_dataset[i : i + bs]
            data_ids = [item["data_id"] for item in batch]
            # image_path = [item["image_path"] for item in batch]
            queries = [item["question"] for item in batch]

            query_vectors = embedder.embed_text(queries)
            print(f"Query vectors shape: {query_vectors.shape}")
            results = indexer.search_batch(query_vectors, top_k=100)

            for data_id, result in zip(data_ids, results):
                retrieved_results.append({"data_id": data_id, "retrieved": result})

        # Save retrieved results
        print(f"Saving retrieval results to {retrieved_results_path}")
        with open(retrieved_results_path, "w") as f:
            json.dump(retrieved_results, f, indent=2)

    # Process retrieved results for evaluation
    retrieved_pairs = {
        item["data_id"]: [
            sub_item["metadata"]["entity_id"] for sub_item in item["retrieved"][2]
        ]
        for item in retrieved_results
    }
    batch_metrics = evaluate_batch(retrieved_pairs, ground_truth_pairs)

    print("\nBatch Metrics:")
    for k, metrics in batch_metrics.items():
        print(f"\nMetrics at k={k}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")


def test():
    # Example usage for testing
    retrieved_results = {
        "infoseek_val_00000000": [
            "Q28324885",
            "Q7075031",
            "Q25054234",
            "Q7075069",
            "Q935215",
        ]
    }

    ground_truth_pairs = {
        "infoseek_val_00000000": "Q934551",
        "infoseek_val_00030546": "Q1121020",
        "infoseek_val_00033267": "Q155869",
    }

    # Calculate metrics
    batch_metrics = evaluate_batch(retrieved_results, ground_truth_pairs)
    print("\nTest Batch Metrics:")
    for k, metrics in batch_metrics.items():
        print(f"\nMetrics at k={k}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
