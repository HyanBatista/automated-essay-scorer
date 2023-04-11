import math


class TfidfVectorizer:
    def __init__(self) -> None:
        self.word_inverse_document_frequency = {}

    def fit(self, documents: list[str]) -> None:
        n_documents = len(documents)
        self._init_word_inverse_document_frequency(documents)
        for term, count in self.word_inverse_document_frequency.items():
            self.word_inverse_document_frequency[term] = math.log(n_documents / count)
    
    def _init_word_inverse_document_frequency(self, documents: list[str]):
        for document in documents:
            seen = set()
            for term in document.split():
                if term not in seen:
                    self.word_inverse_document_frequency.setdefault(term, 0)
                    self.word_inverse_document_frequency[term] += 1
                    seen.add(term)

    def encode(self, documents: list[str]) -> list[list[float]]:
        encodings = []
        for document in documents:
            encoded_document = []
            for term in document.split():
                encoded_term = self.word_inverse_document_frequency[term]
                encoded_document.append(encoded_term)
            encodings.append(encoded_document)
        return encodings
