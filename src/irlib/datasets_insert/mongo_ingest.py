from typing import List, Dict, Optional

from irlib.utils.mongo import get_db


def _normalize_documents(
    documents: List[Dict], collection_name: str
) -> List[Dict]:
    """Μετατρέπει μια λίστα από dicts σε έτοιμα Mongo docs."""
    normalized = []
    for doc in documents:
        # περιμένουμε τουλάχιστον id, text
        doc_id = doc["id"]
        text = doc["text"]

        normalized.append(
            {
                "_id": str(doc_id),
                "collection": collection_name,
                "text": text,
                # εδώ αργότερα μπορείς να βάλεις extra metadata
            }
        )
    return normalized


def _normalize_queries(
    queries: List[Dict], collection_name: str
) -> List[Dict]:
    normalized = []
    for q in queries:
        q_id = q["id"]
        text = q["text"]

        normalized.append(
            {
                "_id": str(q_id),
                "collection": collection_name,
                "text": text,
            }
        )
    return normalized


def _normalize_qrels(
    qrels: List[Dict], collection_name: str
) -> List[Dict]:
    normalized = []
    for qr in qrels:
        normalized.append(
            {
                "collection": collection_name,
                "query_id": str(qr["query_id"]),
                "doc_id": str(qr["doc_id"]),
                "relevance": int(qr.get("relevance", 1)),
            }
        )
    return normalized


def ingest_collection(
    collection_name: str,
    documents: List[Dict],
    queries: List[Dict],
    qrels: List[Dict],
    db_name: Optional[str] = None,
    drop_existing: bool = False,
):
    """
    Εισάγει μια συλλογή IR στη Mongo:

    - Documents -> IR_Lib.Documents
    - Queries   -> IR_Lib.Queries
    - Qrels     -> IR_Lib.Qrels

    collection_name: λογικό όνομα (π.χ. "CF", "NPL", "toy_cf")
    documents: [{ "id": ..., "text": ... }, ...]
    queries:   [{ "id": ..., "text": ... }, ...]
    qrels:     [{ "query_id": ..., "doc_id": ..., "relevance": ... }, ...]
    """
    db = get_db(db_name) if db_name else get_db()

    docs_col = db["Documents"]
    queries_col = db["Queries"]
    qrels_col = db["Qrels"]

    if drop_existing:
        # σβήνουμε ό,τι υπάρχει ήδη για αυτή τη συλλογή
        docs_col.delete_many({"collection": collection_name})
        queries_col.delete_many({"collection": collection_name})
        qrels_col.delete_many({"collection": collection_name})

    docs_to_insert = _normalize_documents(documents, collection_name)
    queries_to_insert = _normalize_queries(queries, collection_name)
    qrels_to_insert = _normalize_qrels(qrels, collection_name)

    if docs_to_insert:
        docs_col.insert_many(docs_to_insert)
    if queries_to_insert:
        queries_col.insert_many(queries_to_insert)
    if qrels_to_insert:
        qrels_col.insert_many(qrels_to_insert)

    return {
        "n_docs": len(docs_to_insert),
        "n_queries": len(queries_to_insert),
        "n_qrels": len(qrels_to_insert),
    }
