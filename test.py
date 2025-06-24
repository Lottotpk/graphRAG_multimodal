import chromadb
print("aaa")
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
print("bbb")
collection = chroma_client.create_collection(name="test")