import argparse
import json
import logging
import joblib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Union

import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

RawJsonMessageCategory = Union[Literal["all_messages"], Literal["spam_messages"]]
RawJsonMessageKey = Union[Literal["uid"], Literal["subject"], Literal["body"]]
RawJsonMessage = Dict[RawJsonMessageKey, str]


class InsufficientTrainingSamplesError(Exception):
    pass


MODEL_VERSION: int = 2


@dataclass(frozen=True, eq=True)
class Model:
    model_version: int
    count_vectorizer: CountVectorizer
    classifier: ComplementNB


WORD_PREPROCESS_RE = re.compile(r"[^a-zA-Z0-9\s]")


def preprocess_message(message: RawJsonMessage) -> str:
    message_str = message["subject"] + " " + message["body"]
    message_str = message_str.lower()
    message_str = WORD_PREPROCESS_RE.sub("", message_str)
    message_str = unicodedata.normalize("NFKD", message_str)
    return message_str


_lemmatizer: WordNetLemmatizer


def init_tokenizer() -> None:
    """Initialize the tokenizer's lemmatizer (hack to get around serializing
    lambdas).
    """
    global _lemmatizer
    nltk.download("wordnet")
    _lemmatizer = WordNetLemmatizer()


def tokenize(text: str) -> np.ndarray:
    global _lemmatizer
    return np.array(
        [_lemmatizer.lemmatize(word) for word in text.split() if 2 <= len(word) <= 20]
    )


def train(
    X: np.ndarray,
    y: np.ndarray,
    shuffle_training_set: bool = True,
) -> Model:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=shuffle_training_set,
    )
    logging.info(
        "Training Bayesian classifier (%d train, %d test)",
        X_train.shape[0],
        X_test.shape[0],
    )
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    assert classifier.class_count_.shape == (2,)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # See https://stackoverflow.com/a/40324184
    confusion = confusion_matrix(y_test, y_pred)
    if confusion.shape != (
        2,
        2,
    ):
        raise InsufficientTrainingSamplesError(
            "Insufficient samples to fully populate confusion matrix (dimensions: {})".format(
                confusion.shape
            )
        )
    false_positive_rate = confusion[0][1] / len(y_test)
    false_negative_rate = confusion[1][0] / len(y_test)
    logging.info(
        "Training complete (accuracy: %f, false positive rate: %f, false negative rate: %f)",
        accuracy,
        false_positive_rate,
        false_negative_rate,
    )

    return classifier


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--messages",
        type=Path,
        required=True,
        help="Path to the processed email messages",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to write the serialized model"
    )
    args = parser.parse_args()
    messages_path: Path = args.messages
    output_path: Path = args.output

    if output_path.exists():
        logging.info("Model already trained, not training again.")
        logging.info(f"(Delete `{output_path}` to re-train.)")
        return

    with open(messages_path) as messages_f:
        messages: Dict[RawJsonMessageCategory, List[RawJsonMessage]] = json.load(
            messages_f
        )

    init_tokenizer()
    count_vectorizer = CountVectorizer(
        # Drop words that only appear in one document.
        min_df=1.5 / len(messages["all_messages"]),
        max_df=0.9,
        preprocessor=preprocess_message,
        tokenizer=tokenize,
    )
    X = count_vectorizer.fit_transform(messages["all_messages"])
    flagged_messages = {m["uid"] for m in messages["spam_messages"]}
    y = np.array([int(m["uid"] in flagged_messages) for m in messages["all_messages"]])
    logging.info(
        "Loaded %d messages, including %d flagged",
        len(y),
        np.count_nonzero(y),
    )

    classifier = train(X, y)
    model = Model(
        model_version=MODEL_VERSION,
        count_vectorizer=count_vectorizer,
        classifier=classifier,
    )

    logging.info("Writing model to %s", output_path)
    with open(args.output, "wb") as model_f:
        joblib.dump(model, model_f)


def load_model(
    model_path: Path,
    import_model: object,
    import_preprocess_message: object,
    import_tokenize: object,
) -> Model:
    # Ensure the caller has imported the required global symbols to unpickle the
    # object.
    _ = (import_model, import_preprocess_message, import_tokenize)

    init_tokenizer()
    with open(model_path, "rb") as f:
        model: Model = joblib.load(f)
    if model.model_version != MODEL_VERSION:
        logging.warn(
            "Serialized model version %d does not match current model version %d",
            model.model_version,
            MODEL_VERSION,
        )
    return model


if __name__ == "__main__":
    main()
