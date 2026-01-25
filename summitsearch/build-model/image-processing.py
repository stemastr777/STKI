import clip
import torch
from PIL import Image
import chromadb
import os
from utils import query_mountain_location

# 1. Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Setup Database
client = chromadb.PersistentClient(path="./mountain_db")
# This creates a "table" for your vectors
collection = client.get_or_create_collection(name="mountain_images")


# print(query_mountain_location("hotaka-dake")["lat"])


def process_and_store(image_path, mountain_data_id):
    # Prepare image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Generate Vector
    with torch.no_grad():
        image_features = model.encode_image(image)
        # Convert to list for the database
        vector = image_features.tolist()[0]

    # Store in ChromaDB
    mountain_name = mountain_data_id.split("_")[1]
    mountain_location = query_mountain_location(mountain_name)
    collection.add(
        embeddings=[vector],
        metadatas=[
            {
                "name": mountain_name,
                "lat": mountain_location["lat"],
                "long": mountain_location["long"],
            }
        ],  # Store the path so you can show the image later
        ids=[mountain_data_id],
    )


# APPLY TO ALL MOUNTAIN IMAGE IN DATA
mountains = os.listdir("../data/")
for mountain in mountains:
    if not os.path.isdir("../data/" + mountain):
        continue

    mountain_dir = f"../data/{mountain}"
    image_files = os.listdir(f"{mountain_dir}/images/")

    print("\n" + mountain.upper())
    for i, image in enumerate(image_files):
        print(image + ":" + f"{i}")
        process_and_store(f"{mountain_dir}/images/{image}", f"img_{mountain}_{i}")
