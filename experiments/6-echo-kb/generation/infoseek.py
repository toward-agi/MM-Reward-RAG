# 把infoseek也挂上吧，这里的结果高的离谱

input_data_file = "/data/dataset/mm_RewardRAG/InfoSeek/valid.json"

import json
import os
from PIL import Image
from tqdm import tqdm

infoseek_data = json.load(open(input_data_file, "r"))

import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.answer_generator.generator_w_knowledge import VLLMAnswerGenerator_w_Knowledge

def convert_wiki_entry_to_text_knowledge(knowledge_entry):
    """
    Convert a Wikipedia entry to a text knowledge format.
    """
    title = knowledge_entry.title
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if (
            "external link" in section_title.lower()
            or "reference" in section_title.lower()
        ):
            continue
        article += (
            "\n## Section Title: "
            + section_title
            + "\n"
            + knowledge_entry.section_texts[it]
        )

    return article

generator = VLLMAnswerGenerator_w_Knowledge()

top_ks = [1, 3, 5, 10, 20, 50, 100]

retrieved_data_path = "/data/track/mm-RewardRAG/experiments/6-echo-kb/retrieve/results/infoseek/retrieved.pkl"
import pickle
retrieved_data = pickle.load(open(retrieved_data_path, "rb"))

assert len(infoseek_data) == len(retrieved_data), "Data length mismatch between infoseek and retrieved data."

for top_k in top_ks:

    output_dir = f"/data/track/mm-RewardRAG/experiments/6-echo-kb/generation/results/infoseek/{top_k}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "answers.jsonl")
    
    batch_size = 2048
    for i in tqdm(range(0, len(infoseek_data), batch_size)):
        batch = infoseek_data[i:i + batch_size]
        retrieved_data_batch = retrieved_data[i:i + batch_size]
        queries = [
            {
                "type": "ti",
                "text": item["question"],
                "image": Image.open(item['image_path']).resize((512, 512)).convert("RGB"),
                "id": item["data_id"],
                "text_documents": [
                    convert_wiki_entry_to_text_knowledge(entry) for entry in retrieved_single_question[:top_k]
                ]
            }
            for item, retrieved_single_question in zip(batch, retrieved_data_batch)
        ]

        answers = generator.generate_answer(queries)
        # write to the output file directly
        with open(output_file, "a") as f:
            for answer in answers:
                f.write(json.dumps(answer) + "\n")
