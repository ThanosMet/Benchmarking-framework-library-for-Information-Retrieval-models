import torch
from sentence_transformers import SentenceTransformer, util
from models.Model import Model
from utilities.document_utls import calc_precision_recall

# Κάνουμε import το εργαλείο της βάσης μας
from irlib.utilities.mongo import get_db


class SBERTModel(Model):
    def __init__(self, collection):
        super().__init__(collection)
        self.model_name = "SBERT"
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self._doc_embeddings = None
        self._weights = []

    def fit(self, min_freq=1, stopwords=True):
        """Υπολογίζει τα Dense Embeddings για έγγραφα και ερωτήματα."""
        print(f"[{self.model_name}] Φόρτωση {len(self.collection.docs)} εγγράφων στο SBERT...")

        # 1. Παίρνουμε τα IDs όλων των εγγράφων
        doc_ids = [str(doc.doc_id) for doc in self.collection.docs]

        # 2. Φέρνουμε τα αυθεντικά κείμενα (raw text) απευθείας από τη MongoDB
        col_name = getattr(self.collection, 'name', 'CRAN')  # Βρίσκουμε ποια συλλογή τρέχουμε
        print(f"[{self.model_name}] Ανάκτηση κειμένων από MongoDB (Συλλογή: {col_name})...")

        db = get_db()
        raw_docs_cursor = db["Documents"].find({"collection": col_name})

        # Φτιάχνουμε ένα λεξικό για αστραπιαία αντιστοίχιση { "ID": "Κείμενο" }
        text_mapping = {str(d["id"]): d["text"] for d in raw_docs_cursor}

        # Χτίζουμε τη λίστα κειμένων με την ΙΔΙΑ ακριβώς σειρά που τα περιμένει το framework
        doc_texts = [text_mapping.get(doc_id, "") for doc_id in doc_ids]

        # 3. Υπολογίζουμε τα Embeddings για τα έγγραφα
        print(f"[{self.model_name}] Υπολογισμός εγγράφων (Encoding)...")
        self._doc_embeddings = self.encoder.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)

        # 4. Παίρνουμε τα κείμενα των queries, ελέγχοντας όλες τις πιθανές μορφές (dict, list, object)
        query_texts = []
        for q in self._queries:
            if isinstance(q, dict):
                q_text = q["text"]
            elif isinstance(q, list):
                q_text = " ".join(q)  # <-- ΕΔΩ ΕΙΝΑΙ Η ΠΡΟΣΘΗΚΗ: Ενώνει τα tokens σε πρόταση!
            else:
                q_text = q.text
            query_texts.append(q_text)

        # 5. Υπολογίζουμε τα Embeddings για τα queries
        print(f"[{self.model_name}] Υπολογισμός {len(query_texts)} queries...")
        query_embeddings = self.encoder.encode(query_texts, convert_to_tensor=True)

        # 6. Υπολογίζουμε το Cosine Similarity (Ομοιότητα Συνημιτόνου)
        print(f"[{self.model_name}] Υπολογισμός Cosine Similarities...")
        cosine_scores = util.cos_sim(query_embeddings, self._doc_embeddings)

        # 7. Μετατρέπουμε τα Tensors σε λίστες από dictionaries
        for i in range(len(query_texts)):
            query_scores = {}
            for j in range(len(doc_ids)):
                score = cosine_scores[i][j].item()
                if score > 0:
                    query_scores[doc_ids[j]] = score
            self._weights.append(query_scores)

        return self

    def evaluate(self, k=None):
        self.precision = []
        self.recall = []

        rel = getattr(self, '_relevant', self.collection.relevant)

        for doc_sim, relevant_docs in zip(self._weights, rel):
            sorted_docs = [doc_id for doc_id, score in sorted(doc_sim.items(), key=lambda item: item[1], reverse=True)]

            cutoff = k if k else len(sorted_docs)
            pre, rec, mrr = calc_precision_recall(sorted_docs, relevant_docs, cutoff)

            self.precision.append(pre)
            self.recall.append(rec)

    def get_model(self):
        return self.model_name

    def _model_func(self, *args, **kwargs):
        pass

    def _vectorizer(self, *args, **kwargs):
        pass