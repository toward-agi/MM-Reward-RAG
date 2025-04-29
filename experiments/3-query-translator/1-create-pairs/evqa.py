# step 0: load the knowledge base and generated queries;

# step 1: use generated queries to retrieve the knowledge base, then calculate the score of the used queries;

# step 2: select the highest score query and lowest score query to generate the DPO training data (pairs);

# step 3: save the pairs to the disk;

import sys
from tqdm import tqdm
sys.path.append("/data/track/mm-RewardRAG")
from core.query_translator.reward_function import score_ranking, extract_json_fields
from core.indexing.faiss_indexer import FaissIndexer
text_indexer = FaissIndexer(mode="bge")
from core.embeddings.bge_text_embedding import BGETextEmbedder
text_embedder = BGETextEmbedder()
text_indexer.load("/data/track/mm-RewardRAG/indexs/bge/encyclopedia/part")
sampled_queries_path = "/data/track/mm-RewardRAG/experiments/3-query-translator/0-sample-queries/results/evqa_results.json"
query_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/evqa_train_1k.json"
output_dir = "/data/track/mm-RewardRAG/experiments/3-query-translator/1-create-pairs/results/"
import os
os.makedirs(output_dir, exist_ok=True)
import json
queries_candiates = json.load(open(sampled_queries_path, "r"))
original_gts = json.load(open(query_path, "r"))
# 这里得写个functions把queries抽取出json string就好了；


selected_queries = []
for queries_group, gt in tqdm(zip(queries_candiates, original_gts)):
    # print(gt)
    queries = queries_group[0]
    scores = []
    for query in tqdm(queries):
        query = extract_json_fields(query)
        query_list = query[0]
        key_words_string = query[1]
        if query_list is None or key_words_string is None:
            scores.append(0.0)
        else:
            query_strings = query_list + [key_words_string]
            # 得到分数之后然后做一个长度归一化就好了；
            query_embeddings = text_embedder.embed_text(query_strings) # 这里不需要归一化吧，都是做好的，在里面；
            retrieval_results = text_indexer.search_batch(query_embeddings, top_k=1000) # 这里一开始的话k可以大一点吧，反正也不保存也不需要那些实体，只需要有entity_list就好了；
            # print(retrieval_results[2])
            entity_lists = [[i["data"]["entity_id"] for i in item[2]] for item in retrieval_results] #我记得是index==1的位置的；之后可以再check一下；不对，返回的只是在metadata里面的id而已，并不是我们想要的entity
            all_scores = [score_ranking([gt["entity_id"]], entity_list) for entity_list in entity_lists]
            num_normalized_scores = sum(all_scores) / len(all_scores) #就这么简单的事情，不会吗？真的是；GPT真笨。
            scores.append(num_normalized_scores)
    
    # 从scores里面选择最大的和最小的分数对应的query；
    max_score_index = scores.index(max(scores))
    min_score_index = scores.index(min(scores))
    max_score_query = queries[max_score_index]
    min_score_query = queries[min_score_index]

    selected_queries.append({
        "max_score_query": max_score_query,
        "min_score_query": min_score_query,
        "max_score": scores[max_score_index],
        "min_score": scores[min_score_index],
    })

with open("/data/track/mm-RewardRAG/experiments/3-query-translator/1-create-pairs/results/evqa_pairs.json", "w") as f:
    json.dump(selected_queries, f, indent=4)
