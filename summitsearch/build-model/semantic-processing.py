from sentence_transformers import SentenceTransformer
import chromadb
import os
from utils import query_mountain_location

# 1. Initialize SBERT (all-mpnet-base-v2 is the most popular/accurate)
model = SentenceTransformer("all-mpnet-base-v2")

# 2. Setup Chroma
client = chromadb.PersistentClient(path="./mountain_db")

# 3. Create a NEW collection specifically for text
text_collection = client.get_or_create_collection(name="mountain_descriptions")


def process_and_store(mountain_data_id):
    mountain_name = mountain_data_id.split("_")[1]

    with open(
        f"../data/{mountain_name}/description.txt", "r", encoding="utf-8"
    ) as file:
        description = file.read()

    # Generate vectors (SBERT does this in one line)
    embeddings = model.encode(description).tolist()

    # Add to the text collection
    mountain_name = mountain_data_id.split("_")[1]
    mountain_location = query_mountain_location(mountain_name)
    text_collection.add(
        ids=mountain_data_id,
        embeddings=embeddings,
        documents=description,  # Storing the actual text here makes it easy to read later
        metadatas=[
            {
                "name": mountain_name,
                "lat": mountain_location["lat"],
                "long": mountain_location["long"],
            }
        ],
    )


# APPLY TO ALL MOUNTAIN DESCRIPTION
mountains = os.listdir("../data/")
for mountain in mountains:
    if not os.path.isdir("../data/" + mountain):
        continue

    print("\n" + mountain.upper())
    mountain_dir = f"../data/{mountain}"
    process_and_store(f"desc_{mountain}")
