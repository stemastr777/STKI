import chromadb

client = chromadb.PersistentClient(path="./mountain_db")

print(client.get_collection(name="mountain_images").get(limit=2))
