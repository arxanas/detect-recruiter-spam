import argparse
import json
import logging
import multiprocessing as mp
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from tqdm import tqdm

RawJsonMessageCategory = Union[Literal["all_messages"], Literal["spam_messages"]]
RawJsonMessageKey = Union[Literal["uid"], Literal["subject"], Literal["body"]]
RawJsonMessage = Dict[RawJsonMessageKey, str]
MessageUid = int
Word = str
WordCount = int
WordCounts = Dict[Word, WordCount]
WordEncoder = List[str]


class InsufficientTrainingSamplesError(Exception):
    pass


@dataclass(frozen=True, eq=True)
class Model:
    word_encoder: WordEncoder
    classifier: ComplementNB


def preprocess_message(message: RawJsonMessage) -> Tuple[MessageUid, WordCounts]:
    lemmatizer = WordNetLemmatizer()

    uid = MessageUid(message["uid"])
    text = message["subject"] + " " + message["body"]
    text = text.replace("\r\n", " ").replace("\n", " ").lower()
    words = text.split()

    def process_word(word: str) -> Optional[str]:
        if len(word) <= 2 or len(word) >= 20:
            return None

        word = word.lower()
        word = lemmatizer.lemmatize(word)
        word = re.sub("[^a-zA-Z]", "", word)
        return word

    word_counts = Counter(
        processed for word in words if (processed := process_word(word)) is not None
    )
    return (uid, word_counts)


def count_words(messages: List[RawJsonMessage]) -> Dict[MessageUid, WordCounts]:
    nltk.download("wordnet")
    uid_to_word_counts = {}
    with mp.Pool() as pool:
        for (uid, word_counts) in tqdm(
            pool.imap(
                preprocess_message,
                messages,
            ),
            desc="Counting words",
            total=len(messages),
        ):
            uid_to_word_counts[uid] = word_counts
    return uid_to_word_counts


def make_word_encoder(all_word_counts: Dict[MessageUid, WordCounts]) -> WordEncoder:
    """Convert all words in the vocabulary into unique label IDs."""
    all_words: Set[str] = set()
    for counts in all_word_counts.values():
        all_words.update(counts.keys())
    return sorted(all_words)


def encode(word_encoder: WordEncoder, word_counts: WordCounts) -> List[int]:
    return [int(encoded_word in word_counts) for encoded_word in word_encoder]


def _make_training(args: Tuple[WordEncoder, WordCounts]) -> List[int]:
    (word_encoder, word_counts) = args
    return encode(word_encoder, word_counts)


def train(
    all_word_counts: Dict[MessageUid, WordCounts],
    flagged_messages: Set[MessageUid],
    shuffle_training_set: bool = True,
) -> Model:
    word_encoder = make_word_encoder(all_word_counts)
    inputs = list(all_word_counts.items())
    classified_messages = [int(uid in flagged_messages) for (uid, _counts) in inputs]
    word_features: List[np.ndarray] = []
    with mp.Pool() as pool:
        for features in tqdm(
            pool.imap(
                _make_training,
                [(word_encoder, counts) for (_uid, counts) in inputs],
                chunksize=500,
            ),
            desc="Creating training data",
            total=len(all_word_counts),
        ):
            word_features.append(np.array(features, copy=True))

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(word_features, copy=True),
        np.array(classified_messages, copy=True),
        test_size=0.2,
        shuffle=shuffle_training_set,
    )
    logging.info(
        "Training Bayesian classifier (%d train, %d test)", len(X_train), len(X_test)
    )
    classifier = ComplementNB()
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

    return Model(
        word_encoder=word_encoder,
        classifier=classifier,
    )


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

    all_word_counts = count_words(messages["all_messages"])
    flagged_messages = {
        MessageUid(message["uid"]) for message in messages["spam_messages"]
    }
    logging.info(
        "Loaded %d messages and %d flagged", len(all_word_counts), len(flagged_messages)
    )
    model: Model = train(all_word_counts, flagged_messages)

    logging.info("Writing model to %s", output_path)
    with open(args.output, "wb") as model_f:
        pickle.dump(model, model_f)


if __name__ == "__main__":
    main()
