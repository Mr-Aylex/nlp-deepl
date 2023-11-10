import keras_core as keras
import keras_nlp
import numpy as np
from keras_core import layers


class F1Score(keras.metrics.F1Score):
    """
    F1 score for sequence classification
    """

    def __init__(self, name='f1_score', dtype=None, average='micro'):
        super().__init__(name=name, dtype=dtype, average=average)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.argmax(y_true, axis=-1)
        y_pred = keras.ops.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


class DumbModel:
    """Dumb model always predict 0"""

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        print("Fitting dumb model")
        self.model.fit(X, y, epochs=10, batch_size=8)

    def predict(self, X):
        return [0] * len(X)

    def dump(self, filename_output):
        pass


class LSTMModel(DumbModel):
    """

    """

    def __init__(self, vocab_size, n_outputs=1, embed_dims=16, encoder_dims=16):
        self.model = keras.Sequential(
            [
                layers.Embedding(vocab_size, embed_dims),
                layers.LSTM(encoder_dims, return_sequences=True),
                layers.TimeDistributed(
                    layers.Dense(n_outputs, activation='softmax') if n_outputs > 1 else layers.Dense(n_outputs,
                                                                                                     activation='sigmoid'))
            ]
        )
        print(self.model(np.random.random((8, 10))).shape)
        print(self.model.summary())
        self.model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                           loss=keras.losses.BinaryCrossentropy())


class GRUModel(DumbModel):
    """

    """

    def __init__(self, vocab_size, n_outputs=1, embed_dims=16, encoder_dims=16):
        self.model = keras.Sequential(
            [
                layers.Embedding(vocab_size, embed_dims),
                layers.GRU(encoder_dims, return_sequences=True),
                layers.TimeDistributed(
                    layers.Dense(n_outputs, activation='softmax') if n_outputs > 1 else layers.Dense(n_outputs,
                                                                                                     activation='sigmoid'))
                # One output by sequence item
            ]
        )
        print(self.model(np.random.random((8, 10))).shape)
        print(self.model.summary())
        self.model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                           loss=keras.losses.BinaryCrossentropy())


class TransformerModel(DumbModel):
    """

    """

    def __init__(self, vocab_size, n_outputs=1, embed_dims=32, encoder_dims=32):
        self.model = keras.Sequential(
            [
                layers.Embedding(vocab_size, embed_dims),
                keras_nlp.layers.TransformerEncoder(encoder_dims, 4),
                keras_nlp.layers.TransformerDecoder(encoder_dims, 4),
                layers.TimeDistributed(
                    layers.Dense(n_outputs, activation='softmax') if n_outputs > 1 else layers.Dense(n_outputs,
                                                                                                     activation='sigmoid'))
                # One output by sequence item
            ]
        )
        print(self.model(np.random.random((8, 10))).shape)
        print(self.model.summary())
        self.model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                           loss=keras.losses.BinaryCrossentropy())


class TextVectoriser(DumbModel):
    def __init__(self, max_len, vocabulary):
        self.model = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocabulary, sequence_length=max_len)
        self.max_len = max_len

    def fit(self, X, y):
        return self

    def transform(self, X):
        out = self.model(X)
        return out

    def predict(self, X):
        out = self.model(X)
        return out

    def dump(self, filename_output):
        pass
