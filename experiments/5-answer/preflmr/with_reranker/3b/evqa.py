retrieved_path = "/data/zhiyuan/flmr/FLMR/examples/EVQA_EVQA_PreFLMR_ViT-G.json"
import json
import pickle


retrieved_data = json.load(open(retrieved_path, "r"))
from datasets import load_dataset

okvqa_question = list(
    load_dataset(
        "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
        "EVQA_data",
    )["test"]
)
knowledge_base = list(
    load_dataset(
        "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
        "EVQA_passages",
    )["test_passages"]
)

id_to_knowledge = {
    item["passage_id"]: item["passage_content"] for item in knowledge_base
}

import sys

sys.path.append("/data/track/mm-RewardRAG")
from core.answer_generator.generator_w_knowledge import VLLMAnswerGenerator_w_Knowledge
from core.reranking.inference_reranker import InferenceReranker

reranker = InferenceReranker(generator_name="vllm_reranker_3b")  # 这里先用3b的模型看看吧；

answer_generator = VLLMAnswerGenerator_w_Knowledge(
    generator_name="vllm_answer_generator_3b"
)
batch_size = 2048

from PIL import Image
from tqdm import tqdm

top_ks = [3, 5, 10, 20, 50, 100]  # 这里之后修改一下，只要把1去掉就好了；
# 然后还有个问题忘记考虑了，就是，如果没有useful/neutral的话，那generator应该怎么处理？就是换个prompt？

for top_k in top_ks:

    output_dir = f"/data/track/mm-RewardRAG/experiments/5-answer/preflmr/results/3b/with_reranker/evqa/{top_k}" # 这里的generator也是3b的；
    import os

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "answers.jsonl")
    reranked_file = os.path.join(output_dir, "reranked.pkl")

    img_dir = "/data/dataset/M2KR_IMG/image/EVQA"

    print(f"Processing top_k: {top_k} reranking...")

    # 这里根本不需要，直接写后面那个就可以了；
    # question_id_to_knowledge_id = {
    #     item["question_id"]: {
    #         "question": item["question"],
    #         "img_path": item["img_path"],
    #         "retrieved_passage": [
    #             id["passage_id"]
    #             for id in retrieved_data[item["question_id"]]["retrieved_passage"][
    #                 :top_k
    #             ]
    #         ],
    #     }
    #     for item in okvqa_question
    # }
    # 把这个转换成那个inference_reranker的输入形式就好了；

    if os.path.exists(reranked_file):
        # logger.info(f"Reranked file already exists, loading from {reranked_file}")
        print(f"Reranked file already exists, loading from {reranked_file}")
        with open(reranked_file, "rb") as f:
            results = pickle.load(f)
        # logger.info(f"Reranking loaded from {reranked_file}")
        print(f"Reranking loaded from {reranked_file}")

    else:
        # logger.info(f"Reranked file not found, reranking from scratch...")
        print(f"Reranked file not found, reranking from scratch...")

        query_doc_list = [
            {
                "question_id": item["question_id"],
                "question": {
                    "type": "ti",
                    "text": item["question"],
                    "image": Image.open(os.path.join(img_dir, item["img_path"]))
                    .resize((512, 512))
                    .convert("RGB"),
                },
                # 这里最好按照单个document来处理，就是一个question_id会有多个doc pair;
                "document": {
                    "type": "t",
                    "text": id_to_knowledge[id["passage_id"]],
                    "document_id": id["passage_id"],
                    # 这里其实是可以添加一个document的id的，就是保存的时候，只保存这个document id就好了；
                }, #直接添加在这个里面算了，这样子的话，就是代码也不用改，到时候直接用rerank之后的结果去分析就好了；
            }
            for item in okvqa_question
            for id in retrieved_data[item["question_id"]]["retrieved_passage"][:top_k]
        ]

        results = reranker.generate(query_doc_list)
        results = reranker.organize_by_question_id(results)
        # 这里保存一下中间结果做为备份，就直接存成pickle文件；
        with open(reranked_file, "wb") as f:
            pickle.dump(results, f)
        # logger.info(f"Reranking done, saved to {reranked_file}")
        print(f"Reranking done, saved to {reranked_file}")

    # logger.info(f"Processing top_k: {top_k} answer generation...")
    print(f"Processing top_k: {top_k} answer generation...")

    for i in tqdm(range(0, len(okvqa_question), batch_size)):
        batch_questions = okvqa_question[i : i + batch_size]
        queries = [
            {
                "type": "ti",
                "text": item["question"],
                "image": Image.open(os.path.join(img_dir, item["img_path"]))
                .resize((512, 512))
                .convert("RGB"),
                "id": item["question_id"],
                # 主要需要修改的点是这里，就是，先把所有的question->doc都拿出来，做一遍rerank，然后把有用的留下；
                "text_documents": [
                    item["text"]
                    for item in results[item["question_id"]]["useful"]
                    + results[item["question_id"]][
                        "neutral"
                    ]  # 这里就先把useful呢neutral都留着吧
                ],
            }
            for item in batch_questions
        ]

        answers = answer_generator.generate_answer(queries)

        # write to the output file directly
        with open(output_file, "a") as f:
            for answer in answers:
                f.write(json.dumps(answer) + "\n")
