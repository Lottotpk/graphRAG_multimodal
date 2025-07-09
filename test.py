from qdrant_client import QdrantClient

if __name__ == "__main__":
    client = QdrantClient(path="qdrant_db/")
    client.create_collection("Test")
