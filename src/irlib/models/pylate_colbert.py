import torch
from time import time
from tqdm import tqdm
from pylate import models

from models.Model import Model
from utilities.document_utls import calc_precision_recall  \


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
                sim_matrix = torch.matmul(q_tensor, d_tensor.T)
                max_sims, _ = torch.max(sim_matrix, dim=1)
                score = torch.sum(max_sims).item()

                rqd[doc.doc_id] = score

            # Append the dictionary of scores
            self._weights.append(rqd)

        return self

    # --------------------------------------------------------
    # Override the Evaluate Method
    # --------------------------------------------------------
    def evaluate(self, k=None):
        """Overrides the parent evaluate to bypass vector/graph requirements."""
        self.precision = []
        self.recall = []

        # Safely get the relevance list
        rel = getattr(self, '_relevant', self.collection.relevant)

        for doc_sim, relevant_docs in zip(self._weights, rel):
            # Sort documents by their ColBERT score (highest first)
            sorted_docs = [doc_id for doc_id, score in sorted(doc_sim.items(), key=lambda item: item[1], reverse=True)]

            # Apply the top-k cutoff
            cutoff = k if k else len(sorted_docs)

            # Calculate metrics using the framework's built-in utility
            pre, rec, mrr = calc_precision_recall(sorted_docs, relevant_docs, cutoff)

            self.precision.append(pre)
            self.recall.append(rec)

    # --------------------------------------------------------
    # Fulfill the Abstract Base Class Requirements
    # --------------------------------------------------------
    def get_model(self):
        return self.model

    def _model_func(self, *args, **kwargs):
        pass

    def _vectorizer(self, *args, **kwargs):
        pass