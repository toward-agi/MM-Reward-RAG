import sys
from pathlib import Path

from numpy import clip

from generators import description
sys.path.append(str(Path(__file__).parent.parent))

import json
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import uvicorn
import argparse

from core.embeddings.clip_embedder import CLIPEmbedder
from core.embeddings.text_embedder import TextEmbedder
from core.indexing.faiss_indexer import FaissIndexer
from generators.description import DescriptionGenerator
from generators.translator import QueryTranslator
from generators.response import ResponseGenerator
from config.config import settings

# Initialize components
clip_embedder = CLIPEmbedder()
text_embedder = TextEmbedder()
mm_image_indexer = FaissIndexer(mode="image")
mm_text_indexer = FaissIndexer(mode="text")
text_indexer = FaissIndexer(mode="text")
#TODO: should be update the number of indexers
description_generator = DescriptionGenerator()
query_translator = QueryTranslator()
response_generator = ResponseGenerator()

class QueryRequest(BaseModel):
    question: str
    use_image: bool = True
    use_text: bool = True

class AddDocumentRequest(BaseModel):
    text: str
    metadata: dict

def main():
    pass

# database
# testset

# mm-rag, reward for 


# doc: <image, text>

# unified embedding
# vit -> projector -> llava / clip CL; image-text; llava: AL
# T@I, I@T: bs = 128, 0.13

# maybe TODO: unified embedding: CLIP,text/img

#  

# stage 0: load datasets
def load_img_dataset(path):
    """
    Load image dataset from a JSON file.
    format:
    [
        {
            "image_id": 0,
            "image_path": "path/to/image.jpg",
            ""metadata": {
        },
        ...
    ]
    """
    return json.load(open(path, "r"))

def load_text_dataset(path):
    """
    Load text dataset from a JSON file.
    format:
    [
        {
            "text_id": 0,
            "text": "text",
            "key",
        },
        ...
    ]
    """
    return json.load(open(path, "r"))

# stage 1: generate description for each image
def generate_descriptions(image_list):
    input_data_for_generation = []
    for item in image_list:
        image = Image.open(item['image_path'])
        image_id = item['image_id']
        input_data_for_generation.append((image, image_id))
    descriptions = description_generator.generate_description_batch(input_data_for_generation)
    for item, description in zip(image_list, descriptions):
        assert item['image_id'] == description[1]
        item['description'] = description[0]
    return image_list

# stage 2: embedding for each images
def embed_mm_images_index(image_list):
    batch_size = 32 # TODO: add to config file
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        images = [Image.open(item['image_path']) for item in batch]
        image_features = clip_embedder.embed_images(images)
        mm_image_indexer.add_vectors(image_features, batch) # add the original 
    return mm_image_indexer

def embed_mm_text_index(image_list):
    batch_size = 32 # TODO: add to config file
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        text_features = text_embedder.embed_text([item['description'] for item in batch])
        mm_text_indexer.add_vectors(text_features, batch)
    return mm_text_indexer

def embed_text_index(text_list):
    batch_size = 32 # TODO: add to config file
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        text_features = text_embedder.embed_text([item['text'] for item in batch])
        text_indexer.add_vectors(text_features, batch)
    return text_indexer

# stage 3: generate query for each question, image pair
"""
testset format:
[
    {
        "question": "question",
        "image_path": "path/to/image.jpg",
        "id": 0
    },
    ...
]
"""
def generate_queries(question_list):
    input_data_for_generation = []
    for item in question_list:
        question = item['question']
        image = Image.open(item['image_path'])
        image_id = item['id']
        input_data_for_generation.append((question, image, image_id))
    queries = query_translator.generate_queries_batch(input_data_for_generation)
    for item, query in zip(question_list, queries):
        assert item['id'] == query[1]
        item['query'] = query[0]
    # TODO: assert query can be parsed
    return question_list


# stage 4: retrieve documents for each query
def retrieve_documents(question_list):
    """Retrieve relevant documents for a list of questions using multimodal and text queries.
    
    Args:
        question_list: List of dictionaries containing queries and image paths
        
    Returns:
        Updated question_list with retrieved documents
    """
    for item in question_list:
        # Parse query to get multimodal and knowledge components
        query = item['query']
        try:
            query_dict = json.loads(query)
            multimodal_query = query_dict['multimodal_query']
            knowledge_query = query_dict['knowledge_query']
        except (json.JSONDecodeError, KeyError):
            # Fallback to using the same query for both if parsing fails
            multimodal_query = query
            knowledge_query = query
            
        # Get multimodal search results
        retrieved_imgs = []
        
        # 1. Search using text query
        text_embedding = clip_embedder.embed_text(multimodal_query)
        img_scores, img_indices, img_meta = mm_image_indexer.search(text_embedding, settings.top_k)
        text_scores, text_indices, text_meta = mm_text_indexer.search(text_embedding, settings.top_k)
        
        # 2. Search using image query if provided
        if 'image_path' in item and item['image_path']:
            try:
                image = Image.open(item['image_path'])
                image_embedding = clip_embedder.embed_images([image])
                img_scores2, img_indices2, img_meta2 = mm_image_indexer.search(image_embedding, settings.top_k)
                
                # Add image search results
                retrieved_imgs.extend([(meta, score) for meta, score in zip(img_meta2, img_scores2)])
            except Exception as e:
                print(f"Error processing image {item['image_path']}: {str(e)}")
        
        # Combine all multimodal search results
        retrieved_imgs.extend([(meta, score) for meta, score in zip(img_meta, img_scores)])
        retrieved_imgs.extend([(meta, score) for meta, score in zip(text_meta, text_scores)])
        
        # Remove duplicates while preserving order based on score
        seen = set()
        unique_imgs = []
        for meta, score in retrieved_imgs:
            # Use a tuple of relevant fields from metadata as hash key
            key = tuple(sorted(meta.items()))
            if key not in seen:
                seen.add(key)
                unique_imgs.append((meta, score))
        
        # Sort by score in descending order (higher is better)
        unique_imgs.sort(key=lambda x: x[1], reverse=True)
        
        # Get text search results
        knowledge_embedding = text_embedder.embed_text(knowledge_query)
        text_scores, text_indices, text_meta = text_indexer.search(knowledge_embedding, settings.top_k)
        retrieved_texts = [(meta, score) for meta, score in zip(text_meta, text_scores)]
        
        # Update item with retrieved documents
        item['retrieved_documents'] = {
            "image": unique_imgs,
            "text": retrieved_texts
        }

    return question_list

"""
testset format:
[
    {
        "question": "question",
        "image_path": "path/to/image.jpg",
        "responses": [
            {
                "response": "response",
                "id": 0 
            },
            ...
        ] 
        "response": "response", #text LVLM: image+text -> text
        "id": 0
        Llava-crtic: image,question,[response] -> rank
        llm-as-a-judge
    },
    ...
]
"""

"""
[
    {
        "id": 0,
        "question": "question",
        "image_path": "path/to/image.jpg",
        "responses": [
            {   
                "response": "response",
                "id": 0
            },
            ...
        ]
        "ranked_responses": [
            {
                "response": "response",
                "id": 0
            },
            ...
        ]
    },
    ...
]
"""

"""
format now:
[
    {
        "id": 0,
        "question": "question",
        "image_path": "path/to/image.jpg",
        "query": "query",
        "json_query": {
            "multimodal_query": "multimodal_query",
            "knowledge_query": "knowledge_query"
        },
        "retrieved_documents": {
            "image": [
                {
                    "image_id": 0,
                    "image_path": "path/to/image.jpg",
                    "description": "description"
                },
                ...
            ],
            "text": [
                {
                    "text_id": 0,
                    "text": "text"
                },
                ...
            ]
        }
    },
    ...
]
"""

# stage 5: generate response for each question
def generate_response(question_list):
    """
    input format:
        [
            (question, mode, image, description, knowledge, id),
            (question, mode, image, description, knowledge, id),
            ...
        ]
    TODO: mode

    modifed input format:
        [   
            (question, mode, image, retrieved_image, retrieved_description, retrieved_knowledge, id),
            (question, mode, image, retrieved_image, retrieved_description, retrieved_knowledge, id),
            ...
        ]

    """
    input_data_for_generation = []
    for item in question_list:
        question = item['question']
        mode = item['mode']
        image = Image.open(item['image_path'])
        retrieved_image = Image.open(item['retrieved_documents']['image'][0]['image_path'])
        retrieved_description = item['retrieved_documents']['image'][0]['description']
        retrieved_knowledge = item['retrieved_documents']['text'][0]['text']
        id = item['id']
        input_data_for_generation.append((question, mode, image, retrieved_image, retrieved_description, retrieved_knowledge, id))
    responses = response_generator.generate_batch(input_data_for_generation)
    for item, response in zip(question_list, responses):
        assert item['id'] == response[1]
        item['response'] = response[0]
    return question_list
