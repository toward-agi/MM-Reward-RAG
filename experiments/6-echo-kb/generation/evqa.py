# 这里就是把结果都TMD的做一遍，然后去测试就好了；
# evqa我不担心，主要是infoseek确实数据集太多太慢了啊；
# 从那个PreFLMR里面把代码copy过来然后处理一下就好了；
# 还是先干完再去吃饭吧

input_data_file = "/data/track/mm-RewardRAG/eval_datasets/encyclopedia/test.json"
"""
{'wikipedia_title': 'Smilax bona-nox',
'wikipedia_url': 'https://en.wikipedia.org/wiki/Smilax_bona-nox',
'question_original': 'What kind of medical usage has this plant?',
'question': 'What kind of medical usage has this plant?',
'question_type': 'templated',
'answer': 'help urinary tract infections and as an antioxidant.|help urinary tract infections and as an antioxidant.|urinary tract infections, antioxidant',
'evidence': 'The roots of this plant are known to help urinary tract infections and as an antioxidant.|The roots of this plant are known to help urinary tract infections and as an antioxidant.|The roots of this plant are known to help urinary tract infections and as an antioxidant.',
'evidence_section_id': '4',
'evidence_section_title': 'Ethnobotany|Ethnobotany|Ethnobotany',
'dataset_name': 'inaturalist',
'dataset_category_id': 6214,
'wikipedia_url_used_in_train': 1,
'encyclopedic_vqa_split': 'test',
'dataset_image_ids': '2715027',
'entity_id': 'entity_1677263',
'question_id': 'test_0',
'dataset_image_id': '2715027',
'image_path': 'iNaturalist2021/val/06214_Plantae_Tracheophyta_Liliopsida_Liliales_Smilacaceae_Smilax_bona-nox/09dd9c22-8cbc-4b19-bbdd-1003986c048c.jpg'}
"""

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

retrieved_data_path = "/data/track/mm-RewardRAG/experiments/6-echo-kb/retrieve/results/evqa/retrieved.pkl"
import pickle
retrieved_data = pickle.load(open(retrieved_data_path, "rb"))

assert len(infoseek_data) == len(retrieved_data), "Data length mismatch between infoseek and retrieved data."

for top_k in top_ks:

    output_dir = f"/data/track/mm-RewardRAG/experiments/6-echo-kb/generation/results/evqa/{top_k}"
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
                "image": Image.open( f"/data/mingyan/encyclopedic/{item['image_path']}").resize((512, 512)).convert("RGB"),
                "id": item["question_id"],
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
