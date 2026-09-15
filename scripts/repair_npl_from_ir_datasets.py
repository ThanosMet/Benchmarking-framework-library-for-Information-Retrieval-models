import sys
import shutil
from pathlib import Path
from collections import defaultdict

import ir_datasets


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from irlib.datasets_insert.mongo_ingest import ingest_collection


NPL_DIR = PROJECT_ROOT / "collections" / "NPL"
NPL_DOCS_DIR = NPL_DIR / "docs"
NPL_QUERIES_FILE = NPL_DIR / "queries.txt"
NPL_QRELS_FILE = NPL_DIR / "relevant.txt"


EXPECTED_DOCS = 11429
EXPECTED_QUERIES = 93
EXPECTED_QRELS = 2083


def main():

    print("=" * 70)
    print("Loading official Vaswani / NPL collection from ir_datasets...")
    print("=" * 70)

    dataset = ir_datasets.load("vaswani")

    # ---------------------------------------------------------
    # 1. Load documents
    # ---------------------------------------------------------
    documents = []

    for doc in dataset.docs_iter():
        documents.append({
            "id": str(doc.doc_id),
            "text": doc.text
        })

    # ---------------------------------------------------------
    # 2. Load queries
    # ---------------------------------------------------------
    queries = []

    for query in dataset.queries_iter():
        queries.append({
            "id": str(query.query_id),
            "text": query.text
        })

    # ---------------------------------------------------------
    # 3. Load qrels
    # ---------------------------------------------------------
    qrels = []

    for qrel in dataset.qrels_iter():
        qrels.append({
            "query_id": str(qrel.query_id),
            "doc_id": str(qrel.doc_id),
            "relevance": int(qrel.relevance)
        })

    print()
    print("Loaded:")
    print(f"  Documents : {len(documents)}")
    print(f"  Queries   : {len(queries)}")
    print(f"  Qrels     : {len(qrels)}")

    # ---------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------
    if len(documents) != EXPECTED_DOCS:
        raise RuntimeError(
            f"Wrong document count: expected {EXPECTED_DOCS}, "
            f"got {len(documents)}"
        )

    if len(queries) != EXPECTED_QUERIES:
        raise RuntimeError(
            f"Wrong query count: expected {EXPECTED_QUERIES}, "
            f"got {len(queries)}"
        )

    if len(qrels) != EXPECTED_QRELS:
        raise RuntimeError(
            f"Wrong qrels count: expected {EXPECTED_QRELS}, "
            f"got {len(qrels)}"
        )

    doc_ids = sorted(int(d["id"]) for d in documents)

    print()
    print(f"Min document ID : {min(doc_ids)}")
    print(f"Max document ID : {max(doc_ids)}")
    print(f"Unique IDs      : {len(set(doc_ids))}")

    if len(set(doc_ids)) != EXPECTED_DOCS:
        raise RuntimeError("Duplicate document IDs detected.")

    # Check that qrels only reference real documents
    actual_doc_ids = {str(d["id"]) for d in documents}

    missing_qrel_docs = sorted({
        qr["doc_id"]
        for qr in qrels
        if qr["doc_id"] not in actual_doc_ids
    })

    if missing_qrel_docs:
        raise RuntimeError(
            f"Qrels reference missing documents. "
            f"Examples: {missing_qrel_docs[:20]}"
        )

    print("\nVALIDATION PASSED.")

    # ---------------------------------------------------------
    # 5. Backup old broken docs directory
    # ---------------------------------------------------------
    if NPL_DOCS_DIR.exists():

        current_files = [
            p for p in NPL_DOCS_DIR.iterdir()
            if p.is_file()
        ]

        backup_dir = NPL_DIR / "docs_partial_backup"

        if len(current_files) != EXPECTED_DOCS:

            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            print(
                f"\nBacking up current incomplete docs "
                f"({len(current_files)} files)..."
            )

            NPL_DOCS_DIR.rename(backup_dir)

        else:
            print("\nExisting docs folder already contains 11429 files.")
            shutil.rmtree(NPL_DOCS_DIR)

    NPL_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 6. Rebuild docs directory
    # ---------------------------------------------------------
    print("\nWriting 11,429 document files...")

    for i, doc in enumerate(documents, start=1):

        doc_path = NPL_DOCS_DIR / str(doc["id"])

        with open(
            doc_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(doc["text"])

        if i % 1000 == 0:
            print(f"  Written {i}/{EXPECTED_DOCS}")

    print(f"  Written {EXPECTED_DOCS}/{EXPECTED_DOCS}")

    # ---------------------------------------------------------
    # 7. Rewrite queries.txt
    # ---------------------------------------------------------
    print("\nWriting queries.txt...")

    with open(
        NPL_QUERIES_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for query in queries:
            text = " ".join(query["text"].split())
            f.write(text + "\n")

    # ---------------------------------------------------------
    # 8. Rewrite relevant.txt
    # One line per query, same format your current parser expects
    # ---------------------------------------------------------
    print("Writing relevant.txt...")

    qrel_map = defaultdict(list)

    for qr in qrels:
        qrel_map[str(qr["query_id"])].append(
            str(qr["doc_id"])
        )

    with open(
        NPL_QRELS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for query in queries:

            qid = str(query["id"])

            relevant_docs = qrel_map.get(qid, [])

            # Sort numeric document IDs
            relevant_docs = sorted(
                relevant_docs,
                key=lambda x: int(x)
            )

            f.write(" ".join(relevant_docs) + "\n")

    # ---------------------------------------------------------
    # 9. Final local-file validation
    # ---------------------------------------------------------
    created_docs = [
        p for p in NPL_DOCS_DIR.iterdir()
        if p.is_file()
    ]

    print()
    print("=" * 70)
    print("LOCAL FILE VALIDATION")
    print("=" * 70)
    print(f"Documents written : {len(created_docs)}")
    print(f"Queries written   : {len(queries)}")
    print(f"Qrels written     : {len(qrels)}")

    if len(created_docs) != EXPECTED_DOCS:
        raise RuntimeError(
            "Wrong number of local NPL document files after rebuild."
        )

    # ---------------------------------------------------------
    # 10. Insert clean collection into MongoDB
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("INSERTING CLEAN NPL COLLECTION INTO MONGODB")
    print("=" * 70)

    result = ingest_collection(
        collection_name="NPL",
        documents=documents,
        queries=queries,
        qrels=qrels,
        drop_existing=True
    )

    print("\nMongo ingest result:")
    print(result)

    print()
    print("=" * 70)
    print("NPL REPAIR COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()