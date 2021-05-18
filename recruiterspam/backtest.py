import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

from .train import (
    MessageUid,
    Model,
    RawJsonMessage,
    RawJsonMessageCategory,
    count_words,
    encode,
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
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model",
    )
    args: argparse.Namespace = parser.parse_args()
    messages_path: Path = args.messages
    model_path: Path = args.model

    with open(messages_path) as messages_f:
        messages: Dict[RawJsonMessageCategory, List[RawJsonMessage]] = json.load(
            messages_f
        )
    all_word_counts = count_words(messages["all_messages"])
    flagged_messages = {
        MessageUid(message["uid"]) for message in messages["spam_messages"]
    }

    with open(model_path, "rb") as f:
        model: Model = pickle.load(f)

    logging.info("The following messages were flagged:")
    for (uid, counts) in all_word_counts.items():
        encoded = encode(model.word_encoder, counts)
        true_flagged = uid in flagged_messages
        pred_flagged = 1 in model.classifier.predict([encoded])
        if pred_flagged:
            message = next(
                m for m in messages["all_messages"] if MessageUid(m["uid"]) == uid
            )
            if true_flagged:
                logging.info("(true positive)  uid=%d %s", uid, message["subject"])
            else:
                logging.info("(false positive) uid=%d %s", uid, message["subject"])


if __name__ == "__main__":
    main()
