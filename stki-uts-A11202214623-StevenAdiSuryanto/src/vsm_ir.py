import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VSMQueryParser:
    def __init__(self, tfidf_matrix, vectorizer, doc_names, k=5):
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        self.doc_names = doc_names
        self.k = k

    def evaluate(self, query: str):
        query = query.strip()
        if not query:
            return []

        query_vec = self.vectorizer.transform([query])  # shape: (1 × terms)

        sim_scores = cosine_similarity(self.tfidf_matrix, query_vec).flatten()

        top_idx = np.argsort(sim_scores)[::-1]

        # Return (doc_name, score)
        results = []
        for i in top_idx:
            score = float(sim_scores[i])
            if score <= 0:
                break
            results.append((self.doc_names[i], score))
            if len(results) == self.k:
                break
        return results
