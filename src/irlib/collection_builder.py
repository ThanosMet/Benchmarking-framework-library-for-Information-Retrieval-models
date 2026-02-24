# src/irlib/collection_builder.py
"""
Bridge: MongoDB List[Dict]  →  Collection object (kalogeropo-compatible)

Φορτώνει μια συλλογή από τη MongoDB και επιστρέφει ένα Collection object
με τα ίδια ακριβώς attributes που περιμένει το GSBModel (και κάθε άλλο Model).

Το GSBModel δεν αλλάζει καθόλου — βλέπει Collection και δουλεύει κανονικά.
"""

import re
from collections import defaultdict
from typing import List, Dict, Optional

from Preprocess.Collection import Collection, update_index
from utilities.document_utls import calculate_tf, remove_punctuation


# ---------------------------------------------------------------------------
# IRDocument — ίδιο interface με Document, χωρίς file I/O
# ---------------------------------------------------------------------------

class IRDocument:
    """
    Μιμείται το Document του kalogeropo χωρίς να διαβάζει αρχείο.
    Έχει τα ίδια ακριβώς attributes:
        .doc_id    (int)
        .terms     (List[str])  — uppercase tokens
        .docs_text (str)
        .tf        (Dict[str, int])
    """

    def __init__(self, doc_id_str: str, text: str) -> None:
        # Εξαγωγή αριθμητικού id — ίδια λογική με Document:
        # int(findall(r'\d+', self.path)[0])
        digits = re.findall(r'\d+', doc_id_str)
        if not digits:
            raise ValueError(f"Δεν βρέθηκε αριθμός στο doc_id '{doc_id_str}'")
        self.doc_id: int = int(digits[0])

        # Tokenization — ίδια λογική με Document.read_document():
        # κείμενο stored ως ένα string στη Mongo → split + uppercase
        tokens = [t.strip().upper()
                  for t in remove_punctuation(text).split()
                  if t.strip()]
        self.terms: List[str] = tokens
        self.docs_text: str = " ".join(tokens)
        self.tf: Dict[str, int] = calculate_tf(tokens)

    def __str__(self) -> str:
        return f"doc ID: {self.doc_id}"


# ---------------------------------------------------------------------------
# Factory function — το κύριο API
# ---------------------------------------------------------------------------

def build_collection_from_mongo(
    collection_name: str,
    stopwords: Optional[List[str]] = None,
    db_name: Optional[str] = None,
) -> Collection:
    """
    Φορτώνει μια IR συλλογή από τη MongoDB και επιστρέφει ένα Collection.

    Args:
        collection_name: Το όνομα της συλλογής (π.χ. "CF", "NPL", "CRAN")
        stopwords:       Προαιρετική custom λίστα stopwords.
                         Αν None, χρησιμοποιείται η default του Collection.
        db_name:         Προαιρετικό override για το όνομα της MongoDB βάσης.

    Returns:
        Collection με συμπληρωμένα:
            .docs, .num_docs, .inverted_index, .queries, .relevant, .stopwords

    Παράδειγμα:
        col = build_collection_from_mongo("CF")
        model = GSBModel(col)
        model.fit(min_freq=1, stopwords=True)
        precision, recall = model.evaluate(k=10)
    """
    from irlib.datasets_insert.mongo_loader import load_collection

    print(f"[collection_builder] Φόρτωση '{collection_name}' από MongoDB...")
    documents, queries, qrels = load_collection(collection_name, db_name)
    print(f"[collection_builder] {len(documents)} docs, {len(queries)} queries, {len(qrels)} qrels")

    # --- Δημιουργία Collection με fake path ώστε να μην τρέξει file I/O ---
    # Το "__mongo__" δεν υπάρχει στο filesystem, οπότε το Collection.__init__
    # απλώς θα εκτυπώσει το path και δεν θα κάνει listdir.
    col = Collection(path="__mongo__", name=collection_name)

    # Καθαρισμός τυχόν υπολειμμάτων από το __init__
    col.docs = []
    col.inverted_index = {}

    # --- 1. Docs + Inverted Index ---
    for d in documents:
        doc = IRDocument(d["id"], d["text"])
        col.docs.append(doc)
        update_index(doc, col.inverted_index)

    # num_docs = max doc_id (ΟΧΙ count — απαιτείται από calculate_tsf)
    col.num_docs = max(doc.doc_id for doc in col.docs) if col.docs else 0
    print(f"[collection_builder] num_docs (max id) = {col.num_docs}")
    print(f"[collection_builder] vocab size = {len(col.inverted_index)}")

    # --- 2. Queries: [["TERM1", "TERM2", ...], ...] ---
    query_ids = [q["id"] for q in queries]
    col.queries = [
        [t.strip().upper()
         for t in remove_punctuation(q["text"]).split()
         if t.strip()]
        for q in queries
    ]

    # --- 3. Relevant: [[int, int, ...], ...] ευθυγραμμισμένο με queries ---
    qrel_map: Dict[str, List[int]] = defaultdict(list)
    for qr in qrels:
        digits = re.findall(r'\d+', str(qr["doc_id"]))
        if digits:
            qrel_map[str(qr["query_id"])].append(int(digits[0]))
        else:
            print(f"  [WARN] Αδύνατη μετατροπή doc_id '{qr['doc_id']}', παραλείπεται.")

    col.relevant = [qrel_map.get(qid, []) for qid in query_ids]

    # Έλεγχος για queries χωρίς relevant
    empty = [query_ids[i] for i, r in enumerate(col.relevant) if not r]
    if empty:
        print(f"  [WARN] {len(empty)} queries χωρίς relevant docs: {empty[:5]}{'...' if len(empty) > 5 else ''}")

    # --- 4. Stopwords override (προαιρετικό) ---
    if stopwords is not None:
        col.stopwords = stopwords

    print(f"[collection_builder] Collection '{collection_name}' έτοιμο.")
    return col
