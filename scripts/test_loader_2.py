from irlib.datasets_insert.mongo_loader import load_collection

if __name__ == "__main__":
    collection_name = "CF"
    docs, queries, qrels = load_collection(collection_name)

    target_id = "01030"   # <-- βάλε εδώ το doc id που ξέρεις

    doc = next((d for d in docs if d.get("id") == target_id), None)

    if doc is None:
        print(f"Doc id={target_id} not found. Example ids:", [d.get("id") for d in docs[:5]])
    else:
        print(f"Found doc {doc['id']}")
        print(doc["text"])          # ή doc["text"][:500] για κόψιμο
