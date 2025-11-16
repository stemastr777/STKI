# docs => {filename: content}
from scipy.sparse import coo_matrix
import re
import preprocess
import numpy as np


def build_incidence_matrix(docs):
    all_words = [word for doc in docs.values() for word in preprocess.tokenize(doc)]
    vocabullary = list(set(all_words))
    num_of_files = len(docs)
    num_of_words = len(vocabullary)

    row_positions = []
    col_positions = []
    for i, (doc, content) in enumerate(docs.items()):
        tokens = preprocess.tokenize(content)
        tokens = set(tokens)
        for token in tokens:
            row_positions.append(vocabullary.index(token))
            col_positions.append(i)

    data = [1] * len(row_positions)
    incidence_matrix = coo_matrix(
        (data, (row_positions, col_positions)), shape=(num_of_words, num_of_files)
    )

    return incidence_matrix


def build_inverted_index(docs):
    inverted_index = {}
    for i, (doc, content) in enumerate(docs.items()):
        position = {}  # {'word': [positions]}
        for index, word in enumerate(preprocess.tokenize(content)):
            if word in position:
                position[word].append(index)
            else:
                position[word] = [index]

        # Fill the index
        for word in position:
            if word in inverted_index:
                inverted_index[word].append((i, len(position[word]), position[word]))
            else:
                inverted_index[word] = [(i, len(position[word]), position[word])]
    return inverted_index


class BooleanQueryParser:
    def __init__(self, matrix: coo_matrix, terms: list[str], doc_ids: list[str]):
        self.matrix = matrix.tocsr()
        self.terms = terms
        self.term_index = {t: i for i, t in enumerate(terms)}
        self.doc_ids = doc_ids
        self.N = len(doc_ids)

    def _get_vector(self, term: str) -> np.ndarray:
        stemmed_term = "".join(preprocess.stems([term.lower()]))
        idx = self.term_index.get(stemmed_term)
        if idx is None:
            return np.zeros(self.N, dtype=np.int8)
        return (
            self.matrix[idx, :].toarray().ravel()
        )  # [0, 1, 1, ..., 0] untuk sebuah word

    def _not(self, vec):
        return 1 - vec

    def _and(self, v1, v2):
        return np.bitwise_and(v1, v2)

    def _or(self, v1, v2):
        return np.bitwise_or(v1, v2)

    def evaluate(self, query: str):
        tokens = re.findall(r"\(|\)|AND|OR|NOT|[A-Za-z0-9_]+", query.upper())

        precedence = {"NOT": 3, "AND": 2, "OR": 1}
        output, stack = [], []

        for token in tokens:
            if token not in precedence:
                token = "".join(preprocess.stems([token.lower()]))
            if token in precedence:
                while (
                    stack
                    and stack[-1] != "("
                    and precedence[stack[-1]] >= precedence[token]
                ):
                    output.append(stack.pop())
                stack.append(token)
            elif token == "(":
                stack.append(token)
            elif token == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                stack.pop()
            else:
                output.append(token)

        while stack:
            output.append(stack.pop())

        # Evaluate postfix
        eval_stack = []
        for token in output:
            if token not in precedence:
                token = "".join(preprocess.stems([token.lower()]))
                eval_stack.append(self._get_vector(token))
            elif token == "NOT":
                v = eval_stack.pop()
                eval_stack.append(self._not(v))
            else:
                v2 = eval_stack.pop()
                v1 = eval_stack.pop()
                if token == "AND":
                    eval_stack.append(self._and(v1, v2))
                elif token == "OR":
                    eval_stack.append(self._or(v1, v2))

        result_vec = eval_stack.pop()
        return [self.doc_ids[i] for i, v in enumerate(result_vec) if v == 1]
