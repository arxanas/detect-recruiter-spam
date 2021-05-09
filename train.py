import json
import logging
import multiprocessing as mp
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import nltk
import numpy as np
from imbox import Imbox
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def download_inbox() -> Path:
    messages_file = Path("messages")
    if messages_file.exists():
        logging.info("Messages already downloaded, not downloading again.")
        logging.info("(Delete `messages` file to re-download.)")
        return messages_file

    app_password = os.environ["GMAIL_APP_PASSWORD"].strip()
    with Imbox(
        "imap.gmail.com",
        username="me@waleedkhan.name",
        password=app_password,
        ssl=True,
        ssl_context=None,
        starttls=False,
    ) as imbox:

        def message_to_json(uid, message) -> Dict[str, Any]:
            return {
                "uid": uid.decode(),
                "subject": message.subject,
                "body": "".join(message.body["plain"]),
                "from": message.sent_from,
            }

        def process_messages(iter) -> Iterable[Dict[str, Any]]:
            for (uid, message) in iter:
                try:
                    logging.info("Processing {}: {}".format(uid, message.subject))
                    yield message_to_json(uid, message)
                except Exception as e:
                    logging.exception(f"Failed to process message: {message!r}", e)

        spam_messages = list(
            process_messages(imbox.messages(folder="all", label="recruiter-spam"))
        )
        all_messages = list(process_messages(imbox.messages(folder="all")))

    data = json.dumps(
        {
            "all_messages": all_messages,
            "spam_messages": spam_messages,
        },
    )
    with open(messages_file, "w") as f:
        f.write(data)
    return messages_file


def preprocess_message(message: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    lemmatizer = WordNetLemmatizer()

    uid = int(message["uid"])
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


def count_words(messages: List[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    nltk.download("wordnet")
    uid_to_word_counts = {}
    with mp.Pool() as pool:
        for (uid, word_counts) in tqdm(
            pool.imap_unordered(
                preprocess_message,
                messages,
            ),
            desc="Counting words",
            total=len(messages),
        ):
            uid_to_word_counts[uid] = word_counts
    return uid_to_word_counts


def make_training(args: Tuple[List[str], Dict[str, int]]) -> List[int]:
    (encoder, word_counts) = args
    return [int(encoded_word in word_counts) for encoded_word in encoder]


def train(
    messages: List[Dict[str, Any]],
    word_counts: Dict[int, Dict[str, int]],
    flagged_messages: Set[int],
) -> ComplementNB:
    # Convert all words in the vocabulary into unique label IDs.
    all_words: Set[str] = set()
    for counts in word_counts.values():
        all_words.update(counts.keys())
    encoder = sorted(all_words)

    inputs = list(word_counts.items())
    classified_messages = [int(uid in flagged_messages) for (uid, _counts) in inputs]
    word_features: List[np.ndarray] = []
    with mp.Pool() as pool:
        for features in tqdm(
            pool.imap(
                make_training,
                [(encoder, counts) for (_uid, counts) in inputs],
                chunksize=500,
            ),
            desc="Creating training data",
            total=len(word_counts),
        ):
            word_features.append(np.array(features, copy=True))

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(word_features, copy=True),
        np.array(classified_messages, copy=True),
        test_size=0.2,
    )
    logging.info(
        "Training Bayesian classifier (%d train, %d test)", len(X_train), len(X_test)
    )
    bayes = ComplementNB()
    bayes.fit(X_train, y_train)
    y_pred = bayes.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Training complete (accuracy: %f)", accuracy)

    logging.info("The following messages were flagged:")
    for (uid, counts) in word_counts.items():
        encoded = make_training((encoder, counts))
        true_flagged = uid in flagged_messages
        pred_flagged = 1 in bayes.predict([encoded])
        if pred_flagged:
            message = next(m for m in messages if int(m["uid"]) == uid)
            if true_flagged:
                logging.info("SUCC %d %s", uid, message["subject"])
            else:
                logging.info("FAIL %d %s", uid, message["subject"])


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    messages_file = download_inbox()
    with open(messages_file) as f:
        messages = json.load(f)

    word_counts = count_words(messages["all_messages"])
    flagged_messages = {int(message["uid"]) for message in messages["spam_messages"]}
    logging.info(
        "Loaded %d messages and %d flagged", len(word_counts), len(flagged_messages)
    )
    bayes = train(messages["all_messages"], word_counts, flagged_messages)


if __name__ == "__main__":
    main()
