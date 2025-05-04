import json
import os
import re
import pickle
from tqdm import tqdm
import sys
import bm25s
import Stemmer

sys.path.append("/data/track/mm-RewardRAG/")

# 文件路径定义
NEW_MMQA_PATH = "/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl"
KB_PATH = "/data/track/mm-RewardRAG/indexs/clip-large/mm_qa/metadata.pkl"
OUTPUT_DIR = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/mmqa/bm25s/q2img_caption"
INDEX_PATH = os.path.join(OUTPUT_DIR, "bm25s_index")
DOC_IDS_PATH = os.path.join(OUTPUT_DIR, "doc_ids.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 辅助函数：清理文本
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# 加载验证数据集
def load_validation_dataset():
    valid_dataset = {}
    with open(NEW_MMQA_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            valid_dataset[item['qid']] = item
    return valid_dataset

valid_dataset = load_validation_dataset()

# 检查是否已经有BM25S索引
index_exists = os.path.exists(INDEX_PATH)
doc_ids_exist = os.path.exists(DOC_IDS_PATH)

# 初始化stemmer
stemmer = Stemmer.Stemmer("english")

# 如果需要创建索引
if not index_exists or not doc_ids_exist:
    print("需要创建新索引或加载文档ID...")
    
    # 加载知识库
    print("正在加载知识库...")
    with open(KB_PATH, 'rb') as f:
        kb_data = pickle.load(f)
    
    # 准备文档集合
    print("准备文档集合...")
    corpus = []
    doc_ids = []
    
    for doc in kb_data:
        # 合并标题和正文以提高检索效果
        text = doc["metadata"]["title"]
        corpus.append(clean_text(text))
        # doc_ids.append(doc.get('snippet_id', doc.get('id', f"doc_{len(doc_ids)}")))
        doc_ids.append(doc["metadata"]["id"])
    
    # 保存文档ID映射
    with open(DOC_IDS_PATH, 'wb') as f:
        pickle.dump(doc_ids, f)
    print(f"已保存 {len(doc_ids)} 个文档ID到 {DOC_IDS_PATH}")
    
    # 分词
    print("正在对语料库进行分词...")
    corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer)
    
    # 创建BM25S索引
    print("创建BM25S索引...")
    retriever = bm25s.BM25(corpus=None, backend='numba', method='lucene')
    retriever.index(corpus_tokens)
    
    # 激活numba以加速检索
    try:
        retriever.activate_numba_scorer()
        print("已激活Numba后端以加速评分。")
    except Exception as e:
        print(f"无法激活Numba: {e}")
    
    # 保存索引
    retriever.save(INDEX_PATH)
    print(f"索引已保存到 {INDEX_PATH}")
else:
    # 加载现有索引
    print(f"正在加载现有索引 {INDEX_PATH}...")
    retriever = bm25s.BM25.load(INDEX_PATH, load_corpus=False)
    
    # 加载文档ID映射
    with open(DOC_IDS_PATH, 'rb') as f:
        doc_ids = pickle.load(f)
    print(f"已加载 {len(doc_ids)} 个文档ID")
    
    # 激活numba以加速检索
    try:
        retriever.activate_numba_scorer()
        print("已激活Numba后端以加速评分。")
    except Exception as e:
        print(f"无法激活Numba: {e}")

# 定义K值
ks = [1, 3, 5, 10, 20, 50, 100]

# 对每个K值评估
for k in ks:
    retrieved_results_file = f"{OUTPUT_DIR}/retrieved_results_{k}.json"
    
    # 检查结果是否已存在
    if not os.path.exists(retrieved_results_file):
        print(f"正在为 k={k} 执行检索...")
        
        id_to_retrieved = {}
        valid_dataset_list = list(valid_dataset.items())
        
        # 分批处理查询以提高效率
        bs = 512
        for i in tqdm(range(0, len(valid_dataset_list), bs)):
            batch = valid_dataset_list[i:i+bs]
            data_ids = [item[0] for item in batch]
            questions = [item[1]["question"] for item in batch]
            types = [item[1]['metadata']['type'] for item in batch]
            supporting_contexts = [item[1]["supporting_context"] for item in batch]
            
            # 对每个查询进行检索
            for data_id, question, supporting_context, type in zip(data_ids, questions, supporting_contexts, types):
                # 查询分词
                query_tokens = bm25s.tokenize(clean_text(question), stemmer=stemmer)
                
                # 执行检索
                docs, scores = retriever.retrieve(query_tokens, k=k)
                
                # 提取检索结果
                retrieved_ids = [doc_ids[int(idx)] for idx in docs[0]]
                retrieved_scores = scores[0].tolist()
                
                # 构建检索结果
                retrieved_docs = []
                for idx, doc_id in enumerate(retrieved_ids):
                    retrieved_docs.append({
                        "id": doc_id,
                        "score": float(retrieved_scores[idx])
                    })
                
                # 存储结果
                id_to_retrieved[data_id] = {
                    "data_id": data_id,
                    "retrieved": [None, None, retrieved_docs],  # 与原格式保持一致
                    "supporting_context": supporting_context,
                    "type": type
                }
        
        # 保存结果
        with open(retrieved_results_file, "w") as f:
            json.dump(id_to_retrieved, f, indent=2)
        print(f"结果已保存到 {retrieved_results_file}")
    else:
        print(f"正在从 {retrieved_results_file} 加载现有结果...")
        id_to_retrieved = json.load(open(retrieved_results_file, "r"))
    
    # 评估召回率
    total_relevant = 0
    total_retrieved_relevant = 0
    query_recalls = []  # 用于计算宏观召回率
    
    # 仅统计TextQ类型的查询
    text_queries_count = 0
    
    for id, item in id_to_retrieved.items():
        type = item["type"]
        # 检查是否包含TextQ
        if "ImageQ" not in type:
            continue
        
        text_queries_count += 1
        retrieved_id = [str(i['id']) for i in item["retrieved"][2]]
        supporting_context_id = [str(i['doc_id']) for i in item["supporting_context"] if i["doc_part"] == "image"]

        # 该查询的相关文档总数
        relevant_count = len(supporting_context_id)
        total_relevant += relevant_count
        
        # 该查询检索到的相关文档数量
        retrieved_relevant_count = 0
        for support_id in supporting_context_id:
            if support_id in retrieved_id:
                retrieved_relevant_count += 1
                total_retrieved_relevant += 1
        
        # 计算该查询的召回率并添加到列表中
        query_recall = retrieved_relevant_count / relevant_count if relevant_count > 0 else 1.0
        query_recalls.append(query_recall)
    
    # 计算微观召回率 (Micro Recall)
    micro_recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
    
    # 计算宏观召回率 (Macro Recall)
    macro_recall = sum(query_recalls) / len(query_recalls) if query_recalls else 0
    
    print(f"Top-{k} Results for TextQ queries ({text_queries_count}):")
    print(f"  Micro-Recall@{k}: {micro_recall:.4f} ({total_retrieved_relevant}/{total_relevant})")
    print(f"  Macro-Recall@{k}: {macro_recall:.4f}")
    print("-" * 50)
