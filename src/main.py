import os
os.environ["KERAS_BACKEND"] = "jax"
import click
import numpy as np
from sklearn.model_selection import cross_val_score
from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model.main import make_model


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task, seq_len=10)
    print("start training")
    model = make_model(max_len=10, vocabulary="model/vocabulary.json")
    print(y.shape)
    model.fit(X, y)

    return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
def compute_vocabulary(input_filename, task):
    from keras_nlp.tokenizers import compute_word_piece_vocabulary
    import tensorflow as tf
    df = make_dataset(input_filename)
    X, y = make_features(df, task, seq_len=10)
    X = tf.data.Dataset.from_tensor_slices(X.tolist())
    compute_word_piece_vocabulary(data=X, vocabulary_output_file="model/vocabulary.json", vocabulary_size=10000)


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)
cli.add_command(compute_vocabulary)

if __name__ == "__main__":
    cli()
