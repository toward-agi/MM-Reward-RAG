from generators.base import BaseGenerator
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

"""
召回的图片的metadata格式如下：不同benchmark不一样，这里是infoseek的格式
{'metadata': {'entity_id': 'Q806563', 'path': '/wikipedia_images_full/Q806/Q806563.jpg'}, 'image_path': '/data/dataset/mm_RewardRAG/InfoSeek/wikipedia_images_full/Q806/Q806563.jpg'}
"""
"""
召回的文本的metadata格式
{'wikidata_id': 'Q16221564', 'wikipedia_title': 'Leaf Huang', 'wikipedia_content': 'Leaf Huang is a Fred Eshelman Distinguished Professor in the UNC Eshelman School of Pharmacy and a Professor UNC/NC State joint Biomedical Engineering department. He has authored and co-authored over 570 peer reviewed articles and as of 2017 carries an h-index of 138.## Biography.Leaf Huang, Ph.D. is the Fred Eshelman Distinguished Professor, in the Eshelman School of Pharmacy, University of North Carolina at Chapel Hill. Dr. Huang’s research has been in the area of gene therapy and targeted drug delivery. He has pioneered the liposome non-viral vector and has designed and manufactured the cationic lipid vector for the first non-viral clinical trial in 1992. He was also the first to publish the activity of polyethylene glycol (PEG) in prolonging the circulation time of liposomes. His current work centers on nanoparticle vectors for gene transfer in tumor and liver. He also continues research in establishing a ligand targeted delivery system for cDNA, mRNA, siRNA, proteins and peptides for tumor growth inhibition and for vaccines in treating cancer and infected diseases.\xa0 He has authored or co-authored more than 600 papers with an H-index of 137. He is also the inventor or co-inventor of 22 US and foreign patents. In 2004, he received the Alec D. Bangham MD FRS Achievement Award, which is the highest honor in liposome research. He was the recipient of the 2013 Distinguished Pharmaceutical Scientist Award which is the highest scientific recognition of the American Association of Pharmaceutical Scientists. He was named a Highly Cited Researcher in “Pharmacology & Toxicology” and then in “Cross Field” each year since 2016. Dr. Huang has also co-founded 6 biotech start-ups in the past.Leaf Huang on North Carolina State UniversityLeaf Huang on Controlled Release Society', 'wikipedia_image_url': None, 'wikipedia_summary': 'Leaf Huang is a Fred Eshelman Distinguished Professor in the UNC Eshelman School of Pharmacy and a Professor UNC/NC State joint Biomedical Engineering department. He has authored and co-authored over 570 peer reviewed articles and as of 2017 carries an h-index of 138.', 'entity_id': 'Q16221564'}
"""

def create_entity_id_to_wikipedia_content(metadata):
    maps = {}
    for item in metadata:
        maps[item["entity_id"]] = item
    return maps

class BaselineLVLMGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("vanilla_lvlm_generator")

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "image" in kwargs and kwargs["image"] is not None
        assert "id" in kwargs and kwargs["id"] is not None
        assert "user_prompt" in kwargs and kwargs["user_prompt"] is not None

        prompts = self.prompts
        messages = [
            # {
            #     "role": "system",
            #     "content": prompts["system"]
            # }, for baseline, system prompt is not necessary
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": kwargs["user_prompt"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._encode_image(kwargs['image'])}"
                        }
                    }
                ]
            }
        ]
        

        # print(messages)
        
        return messages, kwargs["id"]
    
    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        image = kwargs["image"]
        id = kwargs["id"]
        user_prompt = kwargs["user_prompt"]
        messages,id = self._prepare_messages(image=image,id=id,user_prompt=user_prompt)
        return self._generate_single(messages,id)

    def _prepare_messages_for_batch(self, **kwargs) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        assert "image_id_prompts_list" in kwargs and kwargs["image_id_prompts_list"] is not None
        image_id_prompts_list = kwargs["image_id_prompts_list"]
        messages_list = []
        for image, id, user_prompt in tqdm(image_id_prompts_list):
            messages, id = self._prepare_messages(image=image, id=id, user_prompt=user_prompt)
            messages_list.append((messages, id))
        return messages_list

    def generate_description_batch(self, image_id_list: List[Tuple[Image.Image, int, str]]) -> List[Tuple[str, int]]:
        """
        Input format:
        [
            (image: PIL.Image, id, user_prompt),
            ...
        ]
        Output format:
        [
            (description, id),
            ...
        ]
        """
        messages = self._prepare_messages_for_batch(image_id_prompts_list=image_id_list)
        print("prepare done")
        return self.generate_batch(message_list=messages)
    
from core.indexing.faiss_indexer import FaissIndexer
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
import json
from tqdm import tqdm
from PIL import Image

output_dir = "results/infoseek/img2img"
retrieved_results_file = f"{output_dir}/retrieved_image_results.json"
import os
os.makedirs(output_dir, exist_ok=True)
valid_dataset = json.load(open("/data/dataset/mm_RewardRAG/InfoSeek/valid.json", "r"))
if not os.path.exists(retrieved_results_file):
    embedder = JinaClipV2Embedder()
    indexer = FaissIndexer(mode="jina-clip-v2")
    indexer.load("visual_index_gpu/infoseek")
    knowledge_metadata = json.load(open("/data/dataset/mm_RewardRAG/InfoSeek/text_kb.json", "r"))
    id_to_knowledge = create_entity_id_to_wikipedia_content(knowledge_metadata)

    bs = 512
    id_to_retrieved = []
    for i in tqdm(range(0, len(valid_dataset), bs)):
        batch = valid_dataset[i:i+bs]
        data_ids = [item["data_id"] for item in batch]
        questions = [item["image_path"] for item in batch]
        query_vectors = embedder.embed_images(questions)
        print(query_vectors.shape)
        results = indexer.search_batch(query_vectors, top_k=20)
        for data_id, result in zip(data_ids, results):
            # 这里就是，召回图片之后，根据图片的metadata里的entity_id，找到对应的文本知识
            corresponding_text_knowledge = [id_to_knowledge[item["metadata"]["entity_id"]] for item in result[2]]
            id_to_retrieved.append({
                "data_id": data_id,
                "retrieved_image": result,
                "retrieved_knowledge": corresponding_text_knowledge
            })

    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

exit()

def create_knowledge_context(retrieved_result):
    return " ".join([item["wikipedia_content"] for item in retrieved_result])

generator = BaselineLVLMGenerator()

image_id_prompts_list = []
for data, retrieved_item in tqdm(zip(valid_dataset, id_to_retrieved)):
    knowledge_context = create_knowledge_context(retrieved_item["retrieved_knowledge"])
    input_question = data["question"]
    input_question = "External knowledge: " + knowledge_context + " Question: " + input_question
    image = Image.open(data["image_path"]).convert("RGB")
    image_id_prompts_list.append((image, data["data_id"], input_question))

print(image_id_prompts_list[0])

output_file_dir= "results/infoseek/img2img"
import os
os.makedirs(output_file_dir, exist_ok=True)

output_file_path = f"{output_file_dir}/qwen-2-vl.json"
buffer_size = 1000000
for i in tqdm(range(0, len(image_id_prompts_list), buffer_size)):
    buffer = image_id_prompts_list[i:i + buffer_size]
    descriptions = generator.generate_description_batch(buffer)
    with open(output_file_path, "a") as f:
        for description, id in descriptions:
            f.write(json.dumps({
                "data_id": id,
                "description": description
            }) + "\n")

