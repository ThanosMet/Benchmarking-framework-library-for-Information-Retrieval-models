# scripts/insert_cf_collection.py

from pathlib import Path
import os
from typing import List, Dict

from irlib.datasets_insert.mongo_ingest import ingest_collection

# Ρίζα του project, π.χ. .../ir-model-comparison
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Απόλυτο path στα CF docs
CF_DOCS_DIR = PROJECT_ROOT / "collections" / "CF" / "docs"


def load_cf_documents() -> List[Dict]:
    documents: List[Dict] = []

    # Ένα loop ΜΟΝΟ πάνω στα αρχεία του φακέλου
    for doc_path in sorted(CF_DOCS_DIR.iterdir()):
        if not doc_path.is_file():
            continue
        if doc_path.name.startswith("."):
            # αγνόησε κρυφά / περίεργα αρχεία
            continue

        # id = όνομα αρχείου χωρίς επέκταση
        doc_id = doc_path.name  # π.χ. "00001"

        #ΕΔΩ κάνουμε το "ένα token ανά γραμμή" -> "μεγάλο string"
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


if __name__ == "__main__":
    docs = load_cf_documents()
    print(f"Loaded {len(docs)} CF documents from {CF_DOCS_DIR}")

    # DEBUG: έλεγξε ότι όλα τα ids είναι μοναδικά
    ids = [d["id"] for d in docs]
    print(f"Unique ids: {len(set(ids))}")

    result = ingest_collection(
        collection_name="CF",
        documents=docs,
        queries=[],   # προς το παρόν δεν βάζουμε queries
        qrels=[],     # ούτε qrels
        drop_existing=True,  # σβήνει ό,τι CF υπήρχε παλιότερα
    )

    print("Ingest result:", result)
