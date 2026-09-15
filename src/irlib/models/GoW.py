# src/irlib/models/GoW.py
from gowpy.feature_extraction.gow import TwidfVectorizer

from models.Model import Model
from utilities.document_utls import cosine_similarity, calc_precision_recall
from typing import Optional, Any
from numpy import array, ndarray


class Gow(Model):
    """
    Graph-of-Words based information retrieval model using TwidfVectorizer from gowpy.
    """
    def __init__(self,
                 collection,
                 window: int = 4,
                 isdirected: bool = False,
                 min_dfreq: float = 0.0,
                 max_dfreq: float = 1.0,
                 term_weighting_scheme: str = 'degree'):
        self.vectorizer = TwidfVectorizer(
            directed=isdirected,
            window_size=window,
            min_df=min_dfreq,
            max_df=max_dfreq,
            term_weighting=term_weighting_scheme
        )
        super().__init__(collection)

    def get_model(self):
        return self.__class__.__name__

    def _model_func(self, freq_termsets: Any) -> ndarray:
        raise NotImplementedError("Gow model does not implement _model_func directly.")

    def _vectorizer(self, tsf_ij: ndarray, idf: ndarray, *args: Any) -> ndarray:
        raise NotImplementedError("Gow model does not implement _vectorizer directly, use _generate_vectors instead.")

    def _generate_vectors(self, **kwargs) -> tuple[ndarray, ndarray]:
        text = kwargs.get('Text')
        if not text or not isinstance(text, list):
            raise ValueError("Text must be provided as a list of strings.")

        vec = self.vectorizer.fit_transform(text).todense()
        qv = vec[self.collection.num_docs:]
        dv = vec[:self.collection.num_docs]
        return qv, dv

    def fit(self, queries=None, min_freq=None, stopwords=False, *args, **kwargs) -> "Gow":
        if queries is None:
            queries = self._queries

        # Φιλτράρισμα stopwords
        if stopwords:
            queries = [
                [w for w in q if w not in self.collection.stopwords]
                for q in queries
            ]

        if not isinstance(queries, list) or not all(isinstance(q, list) for q in queries):
            raise ValueError("Expected 'queries' to be a list of lists of strings.")

        print(f"[GoW] Building corpus from {len(self.collection.docs)} docs + {len(queries)} queries...")

        # Δημιουργία λίστας με ακριβές μέγεθος για να μην χάνονται τα κενά doc_ids
        text = [""] * self.collection.num_docs

        for doc in self.collection.docs:
            text[doc.doc_id - 1] = " ".join(doc.terms)

        for q in queries:
            text.append(" ".join(q))

        self._queryVectors, self._docVectors = self._generate_vectors(Text=text)
        return self

    def evaluate(self, k=None) -> tuple[ndarray, ndarray]:
        for j, q in enumerate(self._queryVectors):
            eval_list = []
            for i in range(len(self._docVectors)):
                score = cosine_similarity(q, self._docVectors[i, :].transpose())
                eval_list.append((i, float(score)))

            eval_list = sorted(eval_list, key=lambda x: x[1], reverse=True)

            # +1 για να ταιριάζει το index με το πραγματικό doc_id (1-based)
            ordered_docs = [tup[0] + 1 for tup in eval_list]

            self.ranking.append(ordered_docs)
            if k is None:
                k = len(ordered_docs)

            pre, rec, ap, mrr = calc_precision_recall(ordered_docs, self.collection.relevant[j], k)
            self.precision.append(round(pre, 8))
            self.recall.append(round(rec, 8))
            self.average_precision.append(round(ap, 8))
            self.mrr.append(round(mrr, 8))

        return array(self.precision), array(self.recall)