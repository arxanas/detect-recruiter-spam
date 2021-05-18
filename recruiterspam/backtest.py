import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from .train import (
    Model,
    RawJsonMessage,
    RawJsonMessageCategory,
    load_model,
    preprocess_message,
    tokenize,
)


def backtest(
    model: Model, messages: Dict[RawJsonMessageCategory, List[RawJsonMessage]]
) -> Tuple[int, int]:
    X = model.count_vectorizer.transform(messages["all_messages"])
    y = model.classifier.predict(X)
    flagged_uids = {m["uid"] for m in messages["spam_messages"]}

    logging.info("The following messages were flagged:")
    true_positives = 0
    false_positives = 0
    for (i, pred_flagged) in enumerate(y):
        if not pred_flagged:
            continue

        message = messages["all_messages"][i]
        uid = message["uid"]
        true_flagged = bool(uid in flagged_uids)
        if true_flagged:
            true_positives += 1
            logging.info("(true positive)  uid=%s %s", uid, message["subject"])
        else:
            false_positives += 1
            logging.info("(false positive) uid=%s %s", uid, message["subject"])
    return (true_positives, false_positives)


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

    model = load_model(
        model_path,
        import_model=Model,
        import_preprocess_message=preprocess_message,
        import_tokenize=tokenize,
    )
    with open(messages_path) as messages_f:
        messages: Dict[RawJsonMessageCategory, List[RawJsonMessage]] = json.load(
            messages_f
        )

    backtest(model, messages)


if __name__ == "__main__":
    main()
