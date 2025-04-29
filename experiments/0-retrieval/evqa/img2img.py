import json
from tqdm import tqdm
import os
from PIL import Image

def calculate_recall_at_k(qid2groundtruth, image_index_qid2topk, topk=[1, 3]):
    """
    计算 recall@k

    参数:
    - qid2groundtruth: dict，qid到ground truth entity的映射
    - image_index_qid2topk: dict，qid到检索召回的entity的映射
    - topk: list，指定的k值

    返回:
    - recall_at_k: dict，k到对应的recall值
    """
    # 初始化 recall@k 的计数器
    recall_at_k = {k: 0 for k in topk}
    total_qids = len(qid2groundtruth)

    for qid, gt_info in qid2groundtruth.items():
        gt_entity_id = gt_info['entity_id']
        retrieved = image_index_qid2topk.get(qid, [])

        # 提取检索结果的 entity_id，假设已经按顺序排序
        """
        这里是按照list来存储每一个k值的；
        """
        retrieved_entity_ids = [[single_item['entity_id'] for single_item in item] for item in retrieved]

        for k in topk:
            # 获取前 k 个检索结果
            topk_retrieved = retrieved_entity_ids[:k]
            """
            这里就得转换一下，把那个list拆开就好了
            """
            topk_retrieved = [item for sublist in topk_retrieved for item in sublist]
            if gt_entity_id in topk_retrieved:
                recall_at_k[k] += 1

    # 计算 recall@k
    for k in topk:
        recall_at_k[k] = recall_at_k[k] / total_qids if total_qids > 0 else 0.0

    return recall_at_k

def calculate_metrics_for_img_retrieval(retrieved_img_path, ground_truth_path):
    re_img = json.load(open(retrieved_img_path))
    with open(ground_truth_path) as f:
        test = json.load(f)

    qid2groundtruth = {}
    for i in range(len(test)):
        qid2groundtruth[test[i]["question_id"]] = {
            "entity_id": test[i]["entity_id"],
            "section_index": test[i]["evidence_section_id"]
        }
    image_index_qid2top3 = {}
    for i in range(len(re_img)):
        # image_index_qid2top3[re_img[i]["data_id"]] = [item["text_ids"][0] for item in re_img[i]["retrieved_image"][2]]
        image_index_qid2top3[re_img[i]["data_id"]] = [item["text_ids"] for item in re_img[i]["retrieved_image"][2]]
        # 这里一张图片可能对应着多个entity，为什么之前只选择第一个来测试？
        # 显然，这里
    
    recall_at_k = calculate_recall_at_k(qid2groundtruth=qid2groundtruth, image_index_qid2topk=image_index_qid2top3, topk=[1, 3, 5, 10, 20, 50, 100])

    return recall_at_k

"""
上面这些都是在*2img里面一样的，只需要修改后面这些retrieval的函数就可以了。
"""

GT_PATH = "/data/track/mm-RewardRAG/eval_datasets/encyclopedia/test.json"

RETRIEVED_IMG_DIR = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/encyclopedia/jina-clip/img2img"
os.makedirs(RETRIEVED_IMG_DIR, exist_ok=True)
RETRIEVED_IMG_PATH = os.path.join(RETRIEVED_IMG_DIR, "retrieved_img.json")
TEST_IMG_DIR = "/data/mingyan/encyclopedic/"


if os.path.exists(RETRIEVED_IMG_PATH):
    recall_at_k = calculate_metrics_for_img_retrieval(retrieved_img_path=RETRIEVED_IMG_PATH, ground_truth_path=GT_PATH)
    print(recall_at_k)
else:
    import sys
    sys.path.append("/data/track/mm-RewardRAG")
    from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
    from core.indexing.faiss_indexer import FaissIndexer

    embedder = JinaClipV2Embedder()
    indexer = FaissIndexer(mode="jina-clip-v2")
    indexer.load("/data/track/mm-RewardRAG/indexs/jina-clip-v2/visual_index_gpu/encyclopedia/atomic")

    valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/encyclopedia/test.json", "r"))
    """
    这部分在retrieval阶段没有意义；
    """
    # knowledge_metadata = json.load(open("/data/track/mm-RewardRAG/eval_datasets/encyclopedia/wiki.json", "r"))
    # id_to_knowledge = create_entity_id_to_wikipedia_content(knowledge_metadata)

    bs = 512
    id_to_retrieved = []
    for i in tqdm(range(0, len(valid_dataset), bs)):
        batch = valid_dataset[i:i+bs]
        data_ids = [item["question_id"] for item in batch]
        img_paths = [os.path.join(TEST_IMG_DIR, item["image_path"]) for item in batch]
        imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        query_vectors = embedder.embed_images(img_paths)
        print(query_vectors.shape)
        results = indexer.search_batch(query_vectors, top_k=100)
        for data_id, result in zip(data_ids, results):
            id_to_retrieved.append({
                "data_id": data_id,
                "retrieved_image": result,
            })

    with open(RETRIEVED_IMG_PATH, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)

    recall_at_k = calculate_metrics_for_img_retrieval(retrieved_img_path=RETRIEVED_IMG_PATH, ground_truth_path=GT_PATH)
    print(recall_at_k)



    



    