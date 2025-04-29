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
from core.answer_generator.generator_wo_knowledge import VLLMAnswerGenerator_wo_Knowledge

generator = VLLMAnswerGenerator_wo_Knowledge()

output_dir = "/data/track/mm-RewardRAG/experiments/5-answer/wo_knowledge/results/webqa"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "answers.jsonl")

batch_size = 2048
for i in tqdm(range(0, len(infoseek_data), batch_size)):
    batch = infoseek_data[i:i + batch_size]
    queries = [
        {
            "type": "t",
            "text": item["Q"],
            "id": item["Guid"],
        }
        for item in batch
    ]
    answers = generator.generate_answer(queries)
    # write to the output file directly
    with open(output_file, "a") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
