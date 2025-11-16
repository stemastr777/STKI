def precision_at_k(retrieved, relevant):
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    if not retrieved_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(retrieved_set)
