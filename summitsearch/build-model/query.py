from sentence_transformers import SentenceTransformer
import torch
import clip
import chromadb


# 1. Load Models for vectoring user input
sbert_model = SentenceTransformer("all-mpnet-base-v2")
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

# 2. Setup Database
client = chromadb.PersistentClient(path="./mountain_db")
img_collection = client.get_or_create_collection(name="mountain_images")
text_collection = client.get_or_create_collection(name="mountain_descriptions")


def hybrid_search(user_query, n=1):
    # --- STEP A: Search Text (SBERT) ---
    text_vec = sbert_model.encode([user_query]).tolist()
    text_results = text_collection.query(query_embeddings=text_vec, n_results=n)

    # --- STEP B: Search Images (CLIP) ---
    # CLIP needs to tokenize text before encoding
    with torch.no_grad():
        text_tokenized = clip.tokenize([user_query])
        image_query_vec = clip_model.encode_text(text_tokenized).numpy().tolist()

    image_results = img_collection.query(query_embeddings=image_query_vec, n_results=n)

    return text_results, image_results


def get_score_for_mountain(results, target_name):
    """
    results: The dict returned by collection.query()
    target_name: The mountain name we are looking for (e.g., 'Everest')
    """
    # Results are usually in results['metadatas'][0] and results['distances'][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, meta in enumerate(metadatas):
        if meta["name"] == target_name:
            # Convert Distance to Similarity Score (0 to 1)
            # Use 1/(1+dist) for L2 or 1 - (dist/2) for Cosine
            dist = distances[i]
            return 1 / (1 + dist)

    return 0  # Return 0 if the mountain wasn't in the top results


def get_final_ranking(query, n_top=5):
    # 1. Get raw results (Wide net: n_results=10 or 20)
    text_res, img_res = hybrid_search(query, n=20)

    # 2. Extract unique mountain names from BOTH results
    all_candidates = set()
    all_candidates.update([m["name"] for m in text_res["metadatas"][0]])
    all_candidates.update([m["name"] for m in img_res["metadatas"][0]])

    final_scores = []

    for mountain in all_candidates:
        # Get SBERT score for this mountain (or 0 if not in top results)
        t_score = get_score_for_mountain(text_res, mountain)

        # Get BEST CLIP score for this mountain (or 0 if not in top results)
        i_score = get_score_for_mountain(img_res, mountain)

        # Weighted Average (0.5 each)
        # AGREEMENT BOOST: If it has both scores, we can give a 10% bonus
        total = (t_score + i_score) / 2
        if t_score > 0 and i_score > 0:
            total *= 1.1

        final_scores.append({"name": mountain, "score": total})

    # Sort and return
    return sorted(final_scores, key=lambda x: x["score"], reverse=True)[:n_top]


# Execute
query = "A snowy jagged peak under a blue sky"
print(get_final_ranking(query, 3))
