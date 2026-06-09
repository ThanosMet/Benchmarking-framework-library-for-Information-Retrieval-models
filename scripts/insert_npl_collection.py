# scripts/insert_npl_collection.py

from pathlib import Path
import os
from typing import List, Dict

from irlib.datasets_insert.mongo_ingest import ingest_collection

# Ρίζα του project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Απόλυτα paths στα NPL αρχεία
# (Προσάρμοσε τα paths αν τα έβαλες σε διαφορετικούς φακέλους!)
NPL_DOCS_DIR = PROJECT_ROOT / "collections" / "NPL" / "docs"
NPL_QUERIES_FILE = PROJECT_ROOT / "collections" / "NPL"
NPL_QRELS_FILE = PROJECT_ROOT / "collections" / "NPL"

# ---------------------------------------------
# 1. Φόρτωμα NPL documents από τα αρχεία 10005, 10012, ...
# ---------------------------------------------
def load_npl_documents() -> List[Dict]:
    documents: List[Dict] = []

    for doc_path in sorted(NPL_DOCS_DIR.iterdir()):
        if not doc_path.is_file() or doc_path.name.startswith("."):
            continue

        # Το id είναι απλά το όνομα του αρχείου (π.χ. "10005")
        doc_id = doc_path.name

        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            tokens = f.read().split()
            text = " ".join(tokens)

        documents.append(
            {
                "id": doc_id,
                "text": text,
            }
        )

    return documents


# ---------------------------------------------
# 2. Parser για NPL queries.txt και relevant.txt
#    Format:
#    queries.txt  -> 1 query ανά γραμμή
#    relevant.txt -> 1 λίστα από doc_ids ανά γραμμή (ευθυγραμμισμένη με τα queries)
# ---------------------------------------------
def load_npl_queries_and_qrels() -> tuple[List[Dict], List[Dict]]:
    queries: List[Dict] = []
    qrels: List[Dict] = []

    # --- Διάβασμα Queries ---
    with open(NPL_QUERIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        # Αγνοούμε εντελώς κενές γραμμές
        query_lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(query_lines, start=1):
        queries.append({
            "id": str(idx),
            "text": line
        })

    # --- Διάβασμα Qrels (Relevant Docs) ---
    with open(NPL_QRELS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        qrel_lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(qrel_lines, start=1):
        query_id = str(idx)
        # Κόβουμε τη γραμμή με βάση τα κενά για να πάρουμε τα doc_ids
        relevant_docs = line.split()

        for doc_id_str in relevant_docs:
            qrels.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id_str.strip(),
                    # Το NPL χρησιμοποιεί binary relevance, οπότε θεωρούμε "1"
                    # για κάθε έγγραφο που βρίσκεται σε αυτή τη λίστα
                    "relevance": 1,
                }
            )

    return queries, qrels


if __name__ == "__main__":
    docs = load_npl_documents()
    print(f"Loaded {len(docs)} NPL documents from {NPL_DOCS_DIR}")

    ids = [d["id"] for d in docs]
    print(f"Unique doc ids: {len(set(ids))}")

    queries, qrels = load_npl_queries_and_qrels()
    print(f"Loaded {len(queries)} NPL queries and {len(qrels)} qrels.")

    result = ingest_collection(
        collection_name="NPL",
        documents=docs,
        queries=queries,
        qrels=qrels,
        drop_existing=True,
    )

    print("Ingest result:", result)