input_data_file = "/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json"
"""
{
    "data_id": {
        "Q": "",
        "A": [""],
        "topic": "",
        "split": "",
        "Qcate": "",
        "Guid": "",
        "img_posFacts": [
            {
                "image_id": "",
                "title": "",
                "caption: "",
                "url": "",
                "imgUrl": ""
            },
        ],
        "img_negFacts": [
            {
                "image_id": "",
                "title": "",
                "caption: "",
                "url": "",
                "imgUrl": ""
            },
        ],
        "txt_negFacts": [
            {
                "title": "",
                "fact": "",
                "url": "",
                "snippet_id": ""
            },
        ],
        "txt_posFacts": [],
    }
},
"""

import json
import os
from PIL import Image
from tqdm import tqdm

infoseek_data = json.load(open(input_data_file, "r"))
infoseek_data = list(infoseek_data.values())

import sys
sys.path.append("/data/track/mm-RewardRAG/")
# from core.answer_generator.generator_wo_knowledge import VLLMAnswerGenerator_wo_Knowledge
from core.answer_generator.generator_w_gt import VLLMAnswerGenerator_w_GT

# generator = VLLMAnswerGenerator_wo_Knowledge()
generator = VLLMAnswerGenerator_w_GT()

output_dir = "/data/track/mm-RewardRAG/experiments/5-answer/w_gt/results/webqa"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "answers.jsonl")

import pickle

image_kb = pickle.load(open("/data/track/mm-RewardRAG/indexs/clip-large/webqa/metadata.pkl","rb"))
id_to_path = {item["image_id"]: item["path"] for item in image_kb}

batch_size = 2048
for i in tqdm(range(0, len(infoseek_data), batch_size)):
    batch = infoseek_data[i:i + batch_size]
    queries = [
        {
            "type": "t",
            "text": item["Q"],
            "id": item["Guid"],
            "text_documents": [
                doc["fact"]
                for doc in item["txt_posFacts"]
            ],
            "image_documents": [
                {
                    "image": Image.open(id_to_path[str(doc["image_id"])]).convert("RGB").resize((512, 512)),
                    "caption": doc["caption"],
                }
                for doc in item["img_posFacts"]
            ]

        }
        for item in batch
    ]
    answers = generator.generate_answer(queries)
    # write to the output file directly
    with open(output_file, "a") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
