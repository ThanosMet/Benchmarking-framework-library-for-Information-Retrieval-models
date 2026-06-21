from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from models.Model import Model
from utilities.document_utls import calc_precision_recall

# Κάνουμε import το εργαλείο της βάσης μας
from irlib.utilities.mongo import get_db


class LSIModel(Model):
    def __init__(self, collection):
        super().__init__(collection)
        self.model_name = "LSI"
        self._weights = []
        # Χρησιμοποιούμε 100 κρυφές διαστάσεις (Latent Topics)
        self.n_components = 100
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)

    def fit(self, min_freq=1, stopwords=True):
        print(f"[{self.model_name}] Φόρτωση {len(self.collection.docs)} εγγράφων στο LSI...")

        # 1. Παίρνουμε τα IDs όπως είναι στη μνήμη (για να ταιριάζουν με τα Qrels)
        doc_ids = [doc.doc_id for doc in self.collection.docs]

        # 2. Φέρνουμε τα αυθεντικά κείμενα από τη MongoDB
        col_name = getattr(self.collection, 'name', 'CRAN')
        print(f"[{self.model_name}] Ανάκτηση κειμένων από MongoDB (Συλλογή: {col_name})...")

        db = get_db()
        raw_docs_cursor = db["Documents"].find({"collection": col_name})

        def clean_id(raw_id):
            """Ασφαλής καθαρισμός IDs από μηδενικά ('0001' -> '1')"""
            try:
                return str(int(raw_id))
            except ValueError:
                return str(raw_id).strip()

        # Φτιάχνουμε το mapping
        text_mapping = {clean_id(d["id"]): d["text"] for d in raw_docs_cursor}

        # Τραβάμε τα κείμενα (μετατρέποντας το doc_id σε string ΜΟΝΟ για το dictionary lookup)
        doc_texts = [text_mapping.get(clean_id(str(doc_id)), "") for doc_id in doc_ids]

        empty_docs = doc_texts.count("")
        if empty_docs > 0:
            print(f"[ΠΡΟΣΟΧΗ] Το LSI δεν βρήκε κείμενο για {empty_docs} έγγραφα στη βάση!")

        # 3. Βήμα LSI 1: Δημιουργία Πίνακα TF-IDF
        print(f"[{self.model_name}] Δημιουργία TF-IDF Matrix...")
        tfidf_matrix = self.vectorizer.fit_transform(doc_texts)

        # 4. Βήμα LSI 2: Μείωση Διαστάσεων (SVD) για εύρεση κρυφών (Latent) Σημασιολογικών εννοιών
        print(f"[{self.model_name}] Εκτέλεση SVD ({self.n_components} διαστάσεις)...")
        doc_embeddings = self.svd.fit_transform(tfidf_matrix)

        # 5. Διαβάζουμε και καθαρίζουμε τα Queries
        query_texts = []
        for q in self._queries:
            if isinstance(q, dict):
                q_text = q["text"]
            elif isinstance(q, list):
                q_text = " ".join(q)
            else:
                q_text = q.text
            query_texts.append(q_text)

        print(f"[{self.model_name}] Υπολογισμός {len(query_texts)} queries...")
        # 6. Μετατρέπουμε τα queries στον ΊΔΙΟ σημασιολογικό χώρο (TF-IDF -> SVD)
        query_tfidf = self.vectorizer.transform(query_texts)
        query_embeddings = self.svd.transform(query_tfidf)

        # 7. Υπολογίζουμε την Ομοιότητα Συνημιτόνου
        print(f"[{self.model_name}] Υπολογισμός Cosine Similarities...")
        cosine_scores = cosine_similarity(query_embeddings, doc_embeddings)

        # 8. Αποθηκεύουμε τα σκορ
        for i in range(len(query_texts)):
            query_scores = {}
            for j in range(len(doc_ids)):
                score = cosine_scores[i][j]
                if score > 0:
                    query_scores[doc_ids[j]] = score
            self._weights.append(query_scores)

        return self

    def evaluate(self, k=None):
        self.precision = []
        self.recall = []

        rel = getattr(self, '_relevant', self.collection.relevant)

        for doc_sim, relevant_docs in zip(self._weights, rel):
            # Φθίνουσα ταξινόμηση
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