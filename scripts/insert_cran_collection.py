import sys
from pathlib import Path
from typing import List, Dict

# Ρίζα του project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from irlib.datasets_insert.mongo_ingest import ingest_collection

# Απόλυτα paths στα CRAN αρχεία (βάσει του screenshot σου)
CRAN_DOCS_DIR = PROJECT_ROOT / "collections" / "CRAN" / "docs"
CRAN_QUERIES_FILE = PROJECT_ROOT / "collections" / "CRAN" / "Queries.txt"
CRAN_QRELS_FILE = PROJECT_ROOT / "collections" / "CRAN" / "Relevant.txt"


# ---------------------------------------------
# 1. Φόρτωμα CRAN documents
# ---------------------------------------------
def load_cran_documents() -> List[Dict]:
    documents: List[Dict] = []

    for doc_path in sorted(CRAN_DOCS_DIR.iterdir()):
        if not doc_path.is_file() or doc_path.name.startswith("."):
            continue

        # Το doc_path.stem παίρνει το όνομα χωρίς την κατάληξη .txt
        # Το int() αφαιρεί τα μηδενικά μπροστά (π.χ. "0001" -> 1)
        # Το str() το ξανακάνει κείμενο (1 -> "1")
        doc_id = str(int(doc_path.stem))

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
# 2. Parser για CRAN Queries.txt & Relevant.txt
# ---------------------------------------------
def load_cran_queries_and_qrels() -> tuple[List[Dict], List[Dict]]:
    queries: List[Dict] = []
    qrels: List[Dict] = []

    # --- Διάβασμα Queries ---
    with open(CRAN_QUERIES_FILE, "r", encoding="utf-8", errors="ignore") as f:
        query_lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(query_lines, start=1):
        queries.append({
            "id": str(idx),
            "text": line
        })

    # ---Διάβασμα Qrels (Βάσει της πραγματικής δομής του Relevant.txt) ---
    with open(CRAN_QRELS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        # Διαβάζουμε όλες τις γραμμές
        qrel_lines = [line.strip() for line in f if line.strip()]

    # Η Γραμμή 1 (idx=1) έχει τα έγγραφα για το Query 1, κ.ο.κ.
    for q_idx, line in enumerate(qrel_lines, start=1):
        query_id = str(q_idx)

        # Κόβουμε τη γραμμή στα κενά. Κάθε νούμερο είναι ένα σχετικό έγγραφο!
        relevant_docs = line.split()

        for doc_str in relevant_docs:
            doc_id = str(int(doc_str.strip()))

            qrels.append({
                "collection": "CRAN",
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": 1  # Δεν έχουμε βαθμούς 1-4 εδώ, οπότε όλα είναι 1 (Relevant)
            })

    return queries, qrels


if __name__ == "__main__":
    print("Ξεκινάει η ανάγνωση της συλλογής CRAN...")

    docs = load_cran_documents()
    print(f"Διαβάστηκαν {len(docs)} έγγραφα από {CRAN_DOCS_DIR}")

    queries, qrels = load_cran_queries_and_qrels()
    print(f"Διαβάστηκαν {len(queries)} queries και {len(qrels)} σχετικότητες (qrels).")

    print("Εισαγωγή στη MongoDB...")
    result = ingest_collection(
        collection_name="CRAN",
        documents=docs,
        queries=queries,
        qrels=qrels,
        drop_existing=True,
    )

    print("Αποτέλεσμα Ingestion:", result)