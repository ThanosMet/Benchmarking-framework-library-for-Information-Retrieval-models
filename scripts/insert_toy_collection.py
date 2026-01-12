from irlib.datasets.mongo_ingest import ingest_collection
from irlib.utils.mongo import get_db


def build_toy_data():
    # 3 toy documents
    documents = [
        {"id": "D1", "text": "This is a document about information retrieval."},
        {"id": "D2", "text": "MongoDB is a NoSQL database used in many IR systems."},
        {"id": "D3", "text": "This thesis compares multiple information retrieval models."},
    ]

    # 2 toy queries
    queries = [
        {"id": "Q1", "text": "information retrieval models"},
        {"id": "Q2", "text": "mongodb database"},
    ]

    # relevance judgments
    qrels = [
        {"query_id": "Q1", "doc_id": "D1", "relevance": 1},
        {"query_id": "Q1", "doc_id": "D3", "relevance": 1},
        {"query_id": "Q2", "doc_id": "D2", "relevance": 1},
    ]

    return documents, queries, qrels


if __name__ == "__main__":
    collection_name = "toy_demo"

    docs, qs, qrels = build_toy_data()

    stats = ingest_collection(
        collection_name=collection_name,
        documents=docs,
        queries=qs,
        qrels=qrels,
        drop_existing=True,   # να καθαρίζει παλιό toy_demo αν υπάρχει
    )

    print("Inserted:", stats)

    # Διαβάζουμε πίσω από τη Mongo για sanity check
    db = get_db()
    docs_col = db["Documents"]
    queries_col = db["Queries"]
    qrels_col = db["Qrels"]

    print("Docs in Mongo for toy_demo:",
          docs_col.count_documents({"collection": collection_name}))
    print("Queries in Mongo for toy_demo:",
          queries_col.count_documents({"collection": collection_name}))
    print("Qrels in Mongo for toy_demo:",
          qrels_col.count_documents({"collection": collection_name}))
