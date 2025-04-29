import sys
import json
sys.path.append("/data/track/mm-RewardRAG")
import os
import pickle
from core.embeddings.bge_text_embedding import BGETextEmbedder
from core.embeddings.clip_embedder import CLIPEmbedder
from core.indexing.faiss_indexer import FaissIndexer
from PIL import Image
from tqdm import tqdm

sample_data_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/infoseek_train_1k.json"
sampled_data = json.load(open(sample_data_path, "r"))

text_embedder = BGETextEmbedder()
text_indexer = FaissIndexer(mode="bge")
text_indexer.load("/data/track/mm-RewardRAG/indexs/bge/infoseek")

questions = [item["question"] for item in sampled_data]
gt_text = [item['text_metadata']["wikipedia_content"] for item in sampled_data]

question_embeddings = text_embedder.embed_text(questions)
gt_text_embeddings = text_embedder.embed_text(gt_text)

questions_hn = text_indexer.search_batch(question_embeddings, top_k=100)
gt_text_hn = text_indexer.search_batch(gt_text_embeddings, top_k=100)


"""
{'wikidata_id': 'Q16221564', 'wikipedia_title': 'Leaf Huang', 'wikipedia_content': 'Leaf Huang is a Fred Eshelman Distinguished Professor in the UNC Eshelman School of Pharmacy and a Professor UNC/NC State joint Biomedical Engineering department. He has authored and co-authored over 570 peer reviewed articles and as of 2017 carries an h-index of 138.## Biography.Leaf Huang, Ph.D. is the Fred Eshelman Distinguished Professor, in the Eshelman School of Pharmacy, University of North Carolina at Chapel Hill. Dr. Huang’s research has been in the area of gene therapy and targeted drug delivery. He has pioneered the liposome non-viral vector and has designed and manufactured the cationic lipid vector for the first non-viral clinical trial in 1992. He was also the first to publish the activity of polyethylene glycol (PEG) in prolonging the circulation time of liposomes. His current work centers on nanoparticle vectors for gene transfer in tumor and liver. He also continues research in establishing a ligand targeted delivery system for cDNA, mRNA, siRNA, proteins and peptides for tumor growth inhibition and for vaccines in treating cancer and infected diseases.\xa0 He has authored or co-authored more than 600 papers with an H-index of 137. He is also the inventor or co-inventor of 22 US and foreign patents. In 2004, he received the Alec D. Bangham MD FRS Achievement Award, which is the highest honor in liposome research. He was the recipient of the 2013 Distinguished Pharmaceutical Scientist Award which is the highest scientific recognition of the American Association of Pharmaceutical Scientists. He was named a Highly Cited Researcher in “Pharmacology & Toxicology” and then in “Cross Field” each year since 2016. Dr. Huang has also co-founded 6 biotech start-ups in the past.Leaf Huang on North Carolina State UniversityLeaf Huang on Controlled Release Society', 'wikipedia_image_url': None, 'wikipedia_summary': 'Leaf Huang is a Fred Eshelman Distinguished Professor in the UNC Eshelman School of Pharmacy and a Professor UNC/NC State joint Biomedical Engineering department. He has authored and co-authored over 570 peer reviewed articles and as of 2017 carries an h-index of 138.', 'entity_id': 'Q16221564'}
"""

for item, single_question_hn, single_gt_text_hn in tqdm(zip(sampled_data, questions_hn, gt_text_hn),desc="Processing text hard negatives", ):
    question_hd_entity_ids = [item["entity_id"] for item in single_question_hn[2]]
    gt_text_hd_entity_ids = [item["entity_id"] for item in single_gt_text_hn[2]]
    item["hard_negatives"] = {}
    item["hard_negatives"]["input_question"] = question_hd_entity_ids
    item["hard_negatives"]["gt_text"] = gt_text_hd_entity_ids

image_embedder = CLIPEmbedder()
image_indexer = FaissIndexer(mode="clip-large")
image_indexer.load("/data/track/mm-RewardRAG/indexs/clip-large/infoseek")
q_images = [Image.open(item["image_path"]).convert("RGB") for item in sampled_data]
q_image_embeddings = image_embedder.embed_images(q_images)
q_images_hn = image_indexer.search_batch(q_image_embeddings, top_k=100)
for item, single_q_image_hn in tqdm(zip(sampled_data, q_images_hn), desc="Processing image hard negatives"):
    if item["image_metadata"] != "NOT_FOUND":
        gt_image_embedding = image_embedder.embed_images(Image.open(item["image_metadata"]).convert("RGB"))
        gt_image_hn = image_indexer.search(gt_image_embedding, top_k=100)
        gt_image_hd_entity_ids = [item['metadata']["entity_id"] for item in gt_image_hn[2]]
        item["hard_negatives"]["gt_image"] = gt_image_hd_entity_ids
    else:
        item["hard_negatives"]["gt_image"] = "NOT_FOUND"
    q_image_hd_entity_ids = [item['metadata']["entity_id"] for item in single_q_image_hn[2]]
    item["hard_negatives"]["q_image"] = q_image_hd_entity_ids

# Save the results
output_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/infoseek_train_1k_w_hard_negatives.json"

json.dump(sampled_data, open(output_path, "w"), indent=4)

"""
{'metadata': {'entity_id': 'Q806563', 'path': '/wikipedia_images_full/Q806/Q806563.jpg'}, 'image_path': '/data/dataset/mm_RewardRAG/InfoSeek/wikipedia_images_full/Q806/Q806563.jpg'}
"""
