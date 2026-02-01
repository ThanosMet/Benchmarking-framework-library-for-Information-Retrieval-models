from pymongo import MongoClient

# Port 27018 όπως στο docker
MONGO_URI = "mongodb://localhost:27018"


DB_NAME = "IR_Lib"


def get_client() -> MongoClient:
    return MongoClient(MONGO_URI)


def get_db(name: str = DB_NAME):
    client = get_client()
    return client[name]
