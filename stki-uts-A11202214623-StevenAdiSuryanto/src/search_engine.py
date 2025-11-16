#!/usr/bin/env python3
import argparse
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer

import preprocess
from boolean_ir import BooleanQueryParser, build_incidence_matrix
from vsm_ir import VSMQueryParser


def load_preprocess_documents():
    cleaned_docs = {}
    raw_docs = {}
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # folder where search_engine.py lives
    file_folder = os.path.join(base_dir, "..", "data", "*.txt")
    for file in glob.glob(file_folder):
        filename = os.path.basename(file)
        file = open(file, "r", encoding="utf-8")
        text = file.read()

        raw_docs[filename] = text

        cleaned_text = preprocess.clean(text)
        tokens = preprocess.tokenize(cleaned_text)
        tokens_no_stopwords = preprocess.remove_stopwords(tokens)
        tokens_stemmed = preprocess.stems(tokens_no_stopwords)
        cleaned_docs[filename] = " ".join(tokens_stemmed)
    return cleaned_docs


def build_boolean_engine(docs):
    all_words = [word for doc in docs.values() for word in preprocess.tokenize(doc)]
    vocabullary = list(set(all_words))
    incidence_matrix = build_incidence_matrix(docs)
    all_files = list(docs.keys())
    return BooleanQueryParser(incidence_matrix, vocabullary, all_files)


def build_vsm_engine(docs, k):
    names = list(docs.keys())
    contents = list(docs.values())

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), smooth_idf=False)
    tfidf_matrix = vectorizer.fit_transform(contents)

    return VSMQueryParser(tfidf_matrix, vectorizer, names, k=k)


def run_boolean(query, engine):
    results = engine.evaluate(query)
    return results


def run_vsm(query, engine):
    results = engine.evaluate(query)
    return results


def main():
    parser = argparse.ArgumentParser(description="Search Engine Orchestrator")
    parser.add_argument(
        "--model", required=True, choices=["boolean", "vsm"], help="Search model to use"
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k results (VSM only)")
    parser.add_argument(
        "--query", required=True, help='Query string, example: "hutan serangga"'
    )
    args = parser.parse_args()

    # Load docs
    docs = load_preprocess_documents()

    # Pick model
    if args.model == "boolean":
        engine = build_boolean_engine(docs)
        result = run_boolean(args.query, engine)
        print("\n=== BOOLEAN SEARCH RESULTS ===")
        for doc in result:
            print(f"- {doc}")
        print()

    elif args.model == "vsm":
        engine = build_vsm_engine(docs, args.k)
        result = run_vsm(args.query, engine)
        print("\n=== VSM SEARCH RESULTS ===")
        for doc, score in result:
            print(f"- {doc}  (score={score:.4f})")
        print()


if __name__ == "__main__":
    main()
