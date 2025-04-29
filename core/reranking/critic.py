# Standard library imports
import copy
import json
import re
import warnings
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config.config import settings


class LLaVACritic:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.config = settings.get_critic_config("llava_critic")
        self.pretrained = self.config["model"]
        self.model_name = self.config["model_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = self.config["device_map"]
        self.batch_size = self.config["batch_size"]
        self.temperature = self.config["temperature"]
        self.conv_template = self.config["conv_template"]
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.pretrained, None, self.model_name, device_map=self.device_map
        )
        self.model.eval() 
        
    def process_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Process a batch of images"""
        image_tensors = process_images(images, self.image_processor, self.model.config)
        return [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
        
    def get_critic_prompt(self, question: str, response: str) -> str:
        """Generate critic prompt"""
        return f"Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of answer answers provided by a Large Multimodal Model (LMM). Score the response out of 100 and explain your reasoning with specific details. Your output must start with 'Score: X' where X is your score between 0-100. Your task is provided as follows:\nQuestion: [{question}]\nThe LMM response: [{response}]\nASSISTANT:\n "
        
    def get_batch_scores(self, batch_data: List[Dict[str, Any]]) -> List[int]:
        """Get scores for a batch of responses"""
        images = [Image.open(Path(batch_data[0]['data_path']).parent / sample['image_path']) for sample in batch_data]
        image_tensors = self.process_images(images)
        image_sizes = [image.size for image in images]
        
        batch_prompts = []
        for sample in batch_data:
            critic_prompt = self.get_critic_prompt(sample['question'], sample['response']['response'])
            question = DEFAULT_IMAGE_TOKEN + "\n" + critic_prompt
            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            batch_prompts.append(conv.get_prompt())

        # Pad input ids to max length in batch
        tokenized_prompts = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            for prompt in batch_prompts
        ]
        max_length = max(ids.size(0) for ids in tokenized_prompts)
        padded_ids = []
        attention_masks = []
        for ids in tokenized_prompts:
            padding_length = max_length - ids.size(0)
            padded_id = F.pad(ids, pad=(0, padding_length), value=self.tokenizer.pad_token_id)
            attention_mask = F.pad(torch.ones_like(ids), pad=(0, padding_length), value=0)
            padded_ids.append(padded_id)
            attention_masks.append(attention_mask)
        input_ids = torch.cat([ids.unsqueeze(0) for ids in padded_ids]).to(self.device)
        attention_masks = torch.cat([mask.unsqueeze(0) for mask in attention_masks]).to(self.device)

        max_retries, retry_count = 10, 0
        batch_scores = [0] * len(batch_data)
        
        while retry_count < max_retries:
            cont = self.model.generate(
                inputs=input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                modalities=["image"] * len(images),
                attention_mask=attention_masks,
                max_new_tokens=4096,
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            all_scores_found = True
            for i, output in enumerate(text_outputs):
                try:
                    score = re.search(r'Score: (\d+)', output)
                    batch_scores[i] = int(score.group(1))
                except:
                    print(f">>> Could not extract score for sample {text_outputs[i]}")
                    all_scores_found = False
            
            if all_scores_found:
                break
    
            retry_count += 1
            
        if retry_count == max_retries:
            print(">>> Failed to extract some scores after all retries. Defaulting to 0 for failed cases.")
            
        return batch_scores
        
    def rank_responses(self, data_path: str) -> None:
        """Rank responses in data file"""
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Prepare batch data
        all_samples = []

        for sample in data:
            for response in sample['responses']:
                all_samples.append({
                    'id': sample['id'],
                    'data_path': data_path,
                    'image_path': sample['image_path'],
                    'question': sample['question'],
                    'response': response
                })
        
        # Process in batches
        for i in range(0, len(all_samples), self.batch_size):
            batch = all_samples[i:i + self.batch_size]
            batch_scores = self.get_batch_scores(batch)
            # Update scores
            for sample, score in zip(batch, batch_scores):
                sample['response']['score'] = score
        
        # Reorganize results
        results = {}
        for sample in all_samples:
            if sample['id'] not in results:
                results[sample['id']] = []
            results[sample['id']].append({
                "id": sample['response']['id'],
                "score": sample['response']['score'],
                "response": sample['response']['response']
            })
        
        # Sort and update data
        for sample in data:
            ranked_responses = results[sample['id']]
            ranked_responses.sort(key=lambda x: x['score'], reverse=True)
            sample['ranked_responses'] = ranked_responses

        with open(str(Path(data_path).parent / (Path(data_path).stem + '_sorted.json')), 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    judge = LLaVACritic()
    judge.rank_responses('test/critic/data.json')
