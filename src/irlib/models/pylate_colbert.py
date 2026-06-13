import torch
from time import time
from tqdm import tqdm
from pylate import models

from models.Model import Model


class PyLateColBERT(Model):
    def __init__(self, collection, pretrained_model="lightonai/colbertv2.0"):
        super().__init__(collection)
        self.model = self.__class__.__name__

        print(f"Loading PyLate ColBERT model: {pretrained_model}...")
        self.pylate_model = models.ColBERT(model_name_or_path=pretrained_model)

        # Prepare document texts
        self.doc_texts = [doc.docs_text for doc in self.collection.docs]

    def fit(self, min_freq=1, stopwords=True):
        # Note: Neural models don't use apriori min_freq or manual stopwords!

        print("Encoding Documents (This runs incredibly fast)...")
        docs_embeddings = self.pylate_model.encode(
            self.doc_texts,
            batch_size=32,
            is_query=False,
            show_progress_bar=True
        )

        # Convert query token lists back into standard strings
        query_strings = [" ".join(q) for q in self._queries]
        print("Encoding Queries...")
        queries_embeddings = self.pylate_model.encode(
            query_strings,
            batch_size=32,
            is_query=True,
            show_progress_bar=True
        )

        self._weights = []
        print("Scoring Documents via MaxSim...")

        for q_emb in tqdm(queries_embeddings):
            # Convert to PyTorch tensors for fast math
            q_tensor = torch.tensor(q_emb, dtype=torch.float32)
            rqd = {}

            for doc, d_emb in zip(self.collection.docs, docs_embeddings):
                d_tensor = torch.tensor(d_emb, dtype=torch.float32)

                # --- The ColBERT MaxSim Operation ---
                # 1. Calculate similarity matrix between all query tokens and doc tokens
                sim_matrix = torch.matmul(q_tensor, d_tensor.T)

                # 2. Find the maximum similarity for each query token
                max_sims, _ = torch.max(sim_matrix, dim=1)

                # 3. Sum them up to get the final document score
                score = torch.sum(max_sims).item()

                rqd[doc.doc_id] = score

            # Append the dictionary of scores so the parent Model.evaluate() can rank them
            self._weights.append(rqd)

        return self

    def get_model(self):
        return self.model

    def _model_func(self, *args, **kwargs):
        pass

    def _vectorizer(self, *args, **kwargs):
        pass