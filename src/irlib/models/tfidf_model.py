import math
from models.Model import Model
from utilities.document_utls import calc_precision_recall


class TFIDFModel(Model):
    def __init__(self, collection):
        super().__init__(collection)
        self.model = self.__class__.__name__
        self._weights = []

    def fit(self, min_freq=1, stopwords=True):
        """Calculates TF-IDF scores for all documents against all queries."""
        print(f"[{self.model}] Building TF-IDF index for {len(self.collection.docs)} documents...")

        # 1. Calculate Document Frequency (DF) for the entire collection
        # DF is how many distinct documents contain a specific word
        df = {}
        for doc in self.collection.docs:
            # ΑΛΛΑΓΗ ΕΔΩ: Χρησιμοποιούμε τα κλειδιά του doc.tf αντί για doc.tokens
            for term in doc.tf.keys():
                df[term] = df.get(term, 0) + 1

        total_docs = len(self.collection.docs)

        # 2. Score each query
        for query_tokens in self._queries:
            query_scores = {}

            for doc in self.collection.docs:
                score = 0.0

                # TF-IDF calculation for each term in the query
                for term in query_tokens:
                    if term in doc.tf:
                        # Term Frequency (TF): How often the word appears in THIS document
                        tf = doc.tf[term]

                        # Inverse Document Frequency (IDF): How rare the word is across ALL documents
                        # We add 1 to avoid division by zero
                        idf = math.log(total_docs / (df.get(term, 1) + 1))

                        # Accumulate the score
                        score += tf * idf

                # Only store non-zero scores to save memory
                if score > 0:
                    query_scores[doc.doc_id] = score

            self._weights.append(query_scores)

        return self

    def evaluate(self, k=None):
        """Overrides the parent evaluate to use the calculated TF-IDF weights."""
        self.precision = []
        self.recall = []

        rel = getattr(self, '_relevant', self.collection.relevant)

        for doc_sim, relevant_docs in zip(self._weights, rel):
            # Sort documents by their TF-IDF score
            sorted_docs = [doc_id for doc_id, score in sorted(doc_sim.items(), key=lambda item: item[1], reverse=True)]

            cutoff = k if k else len(sorted_docs)
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