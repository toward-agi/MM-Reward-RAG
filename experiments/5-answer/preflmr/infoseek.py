retrieved_path = "/data/zhiyuan/flmr/FLMR/examples/Infoseek_Infoseek_PreFLMR_ViT-G.json"
import json

retrieved_data = json.load(open(retrieved_path, "r"))
from datasets import load_dataset

okvqa_question = list(
    load_dataset(
        "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
        "Infoseek_data",
    )["test"]
)
knowledge_base = list(
    load_dataset(
        "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
        "Infoseek_passages",
    )["test_passages"]
)

id_to_knowledge = {
    item["passage_id"]: item["passage_content"] for item in knowledge_base
}

img_dir = "/data/dataset/M2KR_IMG/image/Infoseek"

import sys
from PIL import Image
from tqdm import tqdm

sys.path.append("/data/track/mm-RewardRAG")
from core.answer_generator.generator_w_knowledge import VLLMAnswerGenerator_w_Knowledge

answer_generator = VLLMAnswerGenerator_w_Knowledge()
batch_size = 2048

top_ks = [1, 3, 5, 10, 20, 50, 100]
for top_k in top_ks:
    # top_k = 5
    output_dir = f"/data/track/mm-RewardRAG/experiments/5-answer/preflmr/results/infoseek/{top_k}"
    import os

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "answers.jsonl")

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
                "text_documents": [
                    id_to_knowledge[id["passage_id"]]
                    for id in retrieved_data[item["question_id"]]["retrieved_passage"][
                        :top_k
                    ]
                ],
            }
            for item in batch_questions
        ]

        answers = answer_generator.generate_answer(queries)

        # write to the output file directly
        with open(output_file, "a") as f:
            for answer in answers:
                f.write(json.dumps(answer) + "\n")
