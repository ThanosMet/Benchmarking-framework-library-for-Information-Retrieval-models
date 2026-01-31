# scripts/insert_cf_collection.py

from pathlib import Path
import os
from typing import List, Dict

from irlib.datasets_insert.mongo_ingest import ingest_collection

# Ρίζα του project, π.χ. .../ir-model-comparison
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Απόλυτο path στα CF docs
CF_DOCS_DIR = PROJECT_ROOT / "collections" / "CF" / "docs"

# Απόλυτο path στο cfquery
CF_QUERY_FILE = PROJECT_ROOT / "datasets" / "CF_RAW" / "cfquery"

# ---------------------------------------------
# 1. Φόρτωμα CF documents από τα αρχεία 00001, 00002, ...
# ---------------------------------------------
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


# ---------------------------------------------
# 2. Parser για cfquery -> queries + qrels
#    Format:
#    QN 00001
#    QU ...
#    NR 00034
#    RD  139 1222  151 2211 ...
# ---------------------------------------------
def load_cf_queries_and_qrels() -> tuple[List[Dict], List[Dict]]:
    queries: List[Dict] = []
    qrels: List[Dict] = []

    current_qid: str | None = None
    current_qtext_lines: List[str] = []
    mode: str | None = None  # "QU" ή "RD" για να ξέρουμε τι συνεχίζουμε

    with open(CF_QUERY_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = raw.strip()

            # κενή γραμμή -> απλά reset mode
            if not stripped:
                mode = None
                continue

            prefix = stripped[:2]

            # Νέο query
            if prefix == "QN":
                # Αν είχαμε προηγούμενο query που δεν έχει γίνει push (σπάνιο),
                # εδώ θα μπορούσαμε να το κάνουμε, αλλά στο CF format
                # το query κλείνει πάντα σε NR, οπότε δεν χρειάζεται.
                parts = stripped.split()
                if len(parts) >= 2:
                    current_qid = parts[1]
                else:
                    current_qid = None
                current_qtext_lines = []
                mode = None

            # Γραμμή που ξεκινάει το κείμενο του query
            elif prefix == "QU":
                mode = "QU"
                # Ό,τι υπάρχει μετά το "QU" είναι το πρώτο κομμάτι του text
                # π.χ. "QU What are the effects ..."
                text_part = " ".join(stripped.split()[1:])
                current_qtext_lines.append(text_part)

            # NR = αριθμός relevant docs (δεν μας νοιάζει για το ingest)
            # εδώ "κλείνουμε" το query text
            elif prefix == "NR":
                if current_qid is not None:
                    text = " ".join(current_qtext_lines).strip()
                    queries.append({"id": current_qid, "text": text})
                mode = "NR"

            # RD = αρχή γραμμών με (doc_id, relevance_code)
            elif prefix == "RD":
                mode = "RD"
                tokens = stripped.split()[1:]  # πετάμε το "RD"
                # ζευγάρια (doc_id, rel_code)
                it = iter(tokens)
                for doc_id_str, rel_str in zip(it, it):
                    # CF docs είναι σε μορφή 00001, 00002, ...
                    doc_id = f"{int(doc_id_str):05d}"
                    qrels.append(
                        {
                            "query_id": current_qid,
                            "doc_id": doc_id,
                            # προς το παρόν κρατάμε τον κωδικό ως int
                            "relevance": int(rel_str),
                        }
                    )

            else:
                # Εδώ έχουμε:
                #  - συνέχεια του QU (γραμμές που αρχίζουν με κενά)
                #  - ή συνέχεια του RD (νέες γραμμές με ζευγάρια αριθμών)
                if mode == "QU" and current_qid is not None:
                    # continuation line του query text
                    current_qtext_lines.append(stripped)
                elif mode == "RD" and current_qid is not None:
                    tokens = stripped.split()
                    it = iter(tokens)
                    for doc_id_str, rel_str in zip(it, it):
                        doc_id = f"{int(doc_id_str):05d}"
                        qrels.append(
                            {
                                "query_id": current_qid,
                                "doc_id": doc_id,
                                "relevance": int(rel_str),
                            }
                        )
                # αλλιώς αγνοούμε τη γραμμή

    return queries, qrels


if __name__ == "__main__":
    docs = load_cf_documents()
    print(f"Loaded {len(docs)} CF documents from {CF_DOCS_DIR}")

    # DEBUG: έλεγξε ότι όλα τα ids είναι μοναδικά
    ids = [d["id"] for d in docs]
    print(f"Unique ids: {len(set(ids))}")

    queries, qrels = load_cf_queries_and_qrels()
    print(f"Loaded {len(queries)} CF queries and {len(qrels)} qrels from {CF_QUERY_FILE}")

    result = ingest_collection(
        collection_name="CF",
        documents=docs,
        queries=queries,   # προς το παρόν δεν βάζουμε queries
        qrels=qrels,     # ούτε qrels
        drop_existing=True,  # σβήνει ό,τι CF υπήρχε παλιότερα
    )

    print("Ingest result:", result)
