import torch
from sentence_transformers import SentenceTransformer, util
from models.Model import Model
from utilities.document_utls import calc_precision_recall


class SBERTModel(Model):
    def __init__(self, collection):
        super().__init__(collection)
        self.model_name = "SBERT"
        # Φορτώνουμε ένα ελαφρύ και γρήγορο προεκπαιδευμένο μοντέλο SBERT
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self._doc_embeddings = None
        self._weights = []

    def fit(self, min_freq=1, stopwords=True):
        """Υπολογίζει τα Dense Embeddings για έγγραφα και ερωτήματα."""
        print(f"[{self.model_name}] Φόρτωση {len(self.collection.docs)} εγγράφων στο SBERT...")

        # 1. Παίρνουμε το πλήρες κείμενο (raw text) όλων των εγγράφων
        doc_texts = [doc.text for doc in self.collection.docs]
        doc_ids = [doc.doc_id for doc in self.collection.docs]

        # 2. Υπολογίζουμε τα Embeddings για τα έγγραφα (αυτό μπορεί να πάρει λίγο χρόνο την πρώτη φορά)
        print(f"[{self.model_name}] Υπολογισμός εγγράφων (Encoding)...")
        # To convert_to_tensor=True μας τα δίνει κατευθείαν σε PyTorch Tensors
        self._doc_embeddings = self.encoder.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)

        # 3. Παίρνουμε τα κείμενα των queries (εξάγουμε το "text" από το dictionary)
        query_texts = [q["text"] if isinstance(q, dict) else q.text for q in self._queries]

        # 4. Υπολογίζουμε τα Embeddings για τα queries
        print(f"[{self.model_name}] Υπολογισμός {len(query_texts)} queries...")
        query_embeddings = self.encoder.encode(query_texts, convert_to_tensor=True)

        # 5. Υπολογίζουμε το Cosine Similarity (Ομοιότητα Συνημιτόνου) ανάμεσα σε ΟΛΑ τα queries και ΟΛΑ τα έγγραφα
        # Το util.cos_sim βγάζει έναν πίνακα μεγέθους: [αριθμός_queries, αριθμός_εγγράφων]
        print(f"[{self.model_name}] Υπολογισμός Cosine Similarities...")
        cosine_scores = util.cos_sim(query_embeddings, self._doc_embeddings)

        # 6. Μετατρέπουμε τα Tensors σε λίστες από dictionaries για να είναι συμβατά με το evaluate()
        for i in range(len(query_texts)):
            query_scores = {}
            for j in range(len(doc_ids)):
                # Παίρνουμε το σκορ και το μετατρέπουμε σε απλό float (από PyTorch Tensor)
                score = cosine_scores[i][j].item()
                if score > 0:  # Κρατάμε μόνο τα θετικά σκορ για εξοικονόμηση μνήμης
                    query_scores[doc_ids[j]] = score
            self._weights.append(query_scores)

        return self

    def evaluate(self, k=None):
        """Χρησιμοποιεί τον ίδιο μηχανισμό ταξινόμησης με το TF-IDF/BM25."""
        self.precision = []
        self.recall = []

        rel = getattr(self, '_relevant', self.collection.relevant)

        for doc_sim, relevant_docs in zip(self._weights, rel):
            # Ταξινομούμε τα έγγραφα με βάση το Cosine Similarity Score (Φθίνουσα σειρά)
            sorted_docs = [doc_id for doc_id, score in sorted(doc_sim.items(), key=lambda item: item[1], reverse=True)]

            cutoff = k if k else len(sorted_docs)
            pre, rec, mrr = calc_precision_recall(sorted_docs, relevant_docs, cutoff)

            self.precision.append(pre)
            self.recall.append(rec)

    # --------------------------------------------------------
    # Υποχρεωτικές μέθοδοι από το Abstract Base Class (Model)
    # --------------------------------------------------------
    def get_model(self):
        return self.model_name

    def _model_func(self, *args, **kwargs):
        pass

    def _vectorizer(self, *args, **kwargs):
        pass