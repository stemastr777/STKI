import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def clean(text):
    text_cleaned = re.sub(re.compile(r"[^a-zA-Z0-9\s]"), "", text)
    text_cleaned = re.sub(re.compile(r"\d"), "", text_cleaned)
    return text_cleaned


def tokenize(text):
    words = word_tokenize(text)
    words = [word for word in words if len(words) > 1]
    words = [word.lower() for word in words]
    return words


def remove_stopwords(tokens):
    Stopwords = set(stopwords.words("indonesian"))
    return [word for word in tokens if word not in Stopwords]


def stems(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in tokens]
