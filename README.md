# Multimodal RAG System

This is a multimodal retrieval-augmented generation system that combines image and text processing capabilities.

## Features

- Image embedding using CLIP
- Image description generation using LVLM
- Text embedding and retrieval
- Query translation and decomposition
- Information fusion with both text and images
- Configurable components (models, reranker, etc.)

## Project Structure

```
.
├── config/
│   ├── config.py           # Configuration classes and utilities
│   └── config.yaml         # Configuration file
├── prompts/
│   ├── description.yaml   # Image description prompts
│   ├── response.yaml      # Response generation prompts
│   ├── translator.yaml    # Query translation prompts
├── core/
│   ├── embeddings/        # Embedding models and utilities dual-encoder
│   ├── indexing/          # Vector store and indexing utilities
│   ├── reranking/         # Reranking models and utilities cross-encoder TODO
├── data/
│   ├── vector_store/     # Vector store data
│   └── images/           # Image storage
├── generators/
│   ├── description.py    # Image description service
│   ├── translator.py     # Query translation service
│   └── response.py       # Final response generation service
├── api/
│   ├── main.py           # main api
└── README.md             # Project README


``` 

## Components

### 1. Embedding Components

#### CLIP
- Purpose: Image embedding generation
- Function: Converts images into vector representations

#### Sentence-Transformers
- Purpose: Text embedding generation
- Function: Converts text into vector representations

### 2. Indexing Components

#### Knowledge Text Indexer
- Purpose: Text database management
- Function: Stores and indexes text knowledge based on text embeddings

#### Multimodal Indexers
1. **Multimodal Image Indexer**
   - Purpose: Image-based search
   - Function: Indexes based on image embeddings

2. **Multimodal Text Indexer**
   - Purpose: Text-based image search
   - Function: Indexes based on image description embeddings

### 3. LVLM-based Generators

#### Description Generator
Purpose: Generate image descriptions for indexing

```json
{
    "image_id": "1",
    "image_path": "path/to/image1.jpg",
    "description": "description1"
}
```

#### Query Translator
Purpose: Decompose input queries for database search

```json
{
    "id": "1",
    "question": "question1",
    "image_path": "path/to/image1.jpg",
    "query": {
        "multimodal_query": "multimodal_query1",
        "text_query": "text_query1"
    }
}
```

#### Response Generator
Purpose: Generate final responses based on retrieved information

```json
{
    "id": "1",
    "question": "question1",
    "image_path": "path/to/image1.jpg",
    "retrieved_documents": {
        "image": [
            {
                "image_id": 0,
                "image_path": "path/to/image.jpg",
                "description": "description"
            }
        ],
        "text": [
            {
                "text_id": 0,
                "text": "text"
            }
        ]
    },
    "responses": [
        {
            "response": "response1",
            "response_id": 0
        }
    ]
}
```

### 4. LLaVA-Critic
Purpose: Response evaluation and ranking

```json
[
    {
        "id": 0,
        "question": "question",
        "image_path": "path/to/image.jpg",
        "responses": [
            {
                "response": "response",
                "response_id": 0
            }
        ],
        "ranked_responses": [
            {
                "response": "response",
                "response_id": 0,
                "score": 0.9
            }
        ]
    }
]
```

## Pipeline

### 1. Embedding & Indexing Process

#### 1.1 Initial Input

##### Image Database
```json
[
    {
        "image_id": "1",
        "image_path": "path/to/image1.jpg"
    }
]
```

##### Text Database
```json
[
    {
        "text_id": "1",
        "text": "text1"
    }
]
```

#### 1.2 Image Description Generation
Process: Generate descriptions for each image

```json
[
    {
        "image_id": "1",
        "image_path": "path/to/image1.jpg",
        "description": "description1"
    }
]
```

#### 1.3 Embedding Generation
1. **Image Database**
   - Generate image embeddings
   - Generate description embeddings

2. **Text Database**
   - Generate text embeddings

### 2. Query Processing

#### Input
- Question text
- Input image

#### Output
```json
{
    "id": "1",
    "question": "question1",
    "image_path": "path/to/image1.jpg",
    "query": {
        "multimodal_query": "multimodal_query1",
        "text_query": "text_query1"
    }
}
```

### 3. Retrieval Process

#### Image Database Retrieval
1. **Multimodal Image Indexer**
   - Uses: multimodal query & image embedding
2. **Multimodal Text Indexer**
   - Uses: multimodal query

#### Text Database Retrieval
- **Knowledge Text Indexer**
   - Uses: text query

#### Output Format
```json
{
    "retrieved_documents": {
        "image": [
            {
                "image_id": 0,
                "image_path": "path/to/image.jpg",
                "description": "description"
            }
        ],
        "text": [
            {
                "text_id": 0,
                "text": "text"
            }
        ]
    }
}
```

### 4. Response Generation
Generates responses based on:
- Input question
- Input image
- Retrieved documents

### 5. Response Evaluation
Uses LLaVA-critic to:
1. Evaluate responses
2. Rank responses by quality
3. Select final response
