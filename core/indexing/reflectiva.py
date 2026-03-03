import os
import faiss
import numpy as np
import torch
import ujson

class Retriever:
    def __init__(self, index_path, index_path_json):

        print("Loading retrieval index")
        self.index = faiss.read_index(index_path)
        self.values = ujson.load(open(index_path_json), 'r')
        print("Done loading retrieval index")
    
    def retrieve(self, query, k):
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = query.astype(np.float32)
        _, indexes = self.index.search(query, k=100)
        chosen_k = indexes[0, :k].tolist()
        return [self.values[k][0] for k in chosen_k]
    
    def retrieve_batch(self, queries, k):

        if isinstance(queries, torch.Tensor):
            queries = queries.detach().cpu().numpy()
        queries = queries.astype(np.float32)
        
        _, indexes = self.index.search(queries, k=100)
        
        results = []
        for i in range(len(queries)):
            chosen_k = indexes[i, :k].tolist()
            results.append([self.values[idx][0] for idx in chosen_k])
        
        return results
