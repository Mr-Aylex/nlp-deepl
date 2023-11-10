from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from src.model.dumb_model import TextVectoriser, LSTMModel, TransformerModel


def make_model(max_len, vocabulary):

    vect = TextVectoriser(max_len=max_len, vocabulary=vocabulary)
    voc_size = vect.model.vocabulary_size()

    return Pipeline([
        ("vectorizer", vect),
        ("model", TransformerModel(voc_size))
    ])
