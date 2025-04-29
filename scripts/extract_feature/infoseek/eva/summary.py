# from core.embeddings.clip_embedder import CLIPEmbedder
import sys
sys.path.append("/data/track/mm-RewardRAG/")
# from core.embeddings.bge_text_embedding import BGETextEmbedder
from core.embeddings.eva_clip import EVAEmbedder

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

text_embedder = EVAEmbedder()
"""
{
        "wikidata_id": "Q16221564",
        "wikipedia_title": "Leaf Huang",
        "wikipedia_content": "Leaf Huang is a Fred Eshelman Distinguished Professor in the UNC Eshelman School of Pharmacy and a Professor UNC/NC State joint Biomedical Engineering department. He has authored and co-authored over 570 peer reviewed articles and as of 2017 carries an h-index of 138.## Biography.Leaf Huang, Ph.D. is the Fred Eshelman Distinguished Professor, in the Eshelman School of Pharmacy, University of North Carolina at Chapel Hill. Dr. Huang\u2019s research has been in the area of gene therapy and targeted drug delivery. He has pioneered the liposome non-viral vector and has designed and manufactured the cationic lipid vector for the first non-viral clinical trial in 1992. He was also the first to publish the activity of polyethylene glycol (PEG) in prolonging the circulation time of liposomes. His current work centers on nanoparticle vectors for gene transfer in tumor and liver. He also continues research in establishing a ligand targeted delivery system for cDNA, mRNA, siRNA, proteins and peptides for tumor growth inhibition and for vaccines in treating cancer and infected diseases.\u00a0 He has authored or co-authored more than 600 papers with an H-index of 137. He is also the inventor or co-inventor of 22 US and foreign patents. In 2004, he received the Alec D. Bangham MD FRS Achievement Award, which is the highest honor in liposome research. He was the recipient of the 2013 Distinguished Pharmaceutical Scientist Award which is the highest scientific recognition of the American Association of Pharmaceutical Scientists. He was named a Highly Cited Researcher in \u201cPharmacology & Toxicology\u201d and then in \u201cCross Field\u201d each year since 2016. Dr. Huang has also co-founded 6 biotech start-ups in the past.Leaf Huang on North Carolina State UniversityLeaf Huang on Controlled Release Society",
        "wikipedia_image_url": null,
        "wikipedia_summary": "Leaf Huang is a Fred Eshelman Distinguished Professor in the UNC Eshelman School of Pharmacy and a Professor UNC/NC State joint Biomedical Engineering department. He has authored and co-authored over 570 peer reviewed articles and as of 2017 carries an h-index of 138.",
        "entity_id": "Q16221564"
    },
"""

input_data_file = "/data/dataset/mm_RewardRAG/InfoSeek/text_kb.json"
text_feature_dir = "/data/dataset/mm_RewardRAG/InfoSeek/eva/textual_summary_features"

text_data = json.load(open(input_data_file, "r"))

# Embed text
batch_size = 1024

os.makedirs(text_feature_dir, exist_ok=True)

for i in tqdm(range(0, len(text_data), batch_size)):
    batch = text_data[i:i + batch_size]
    texts = [item["wikipedia_summary"] for item in batch]
    image_features = text_embedder.embed_text(texts)

    batch_file = os.path.join(text_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized
