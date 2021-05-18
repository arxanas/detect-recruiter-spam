import argparse
import logging
import pickle
import sys
from pathlib import Path

from .train import Model, count_words, encode, preprocess_message


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model",
    )
    args: argparse.Namespace = parser.parse_args()
    model_path: Path = args.model

    with open(model_path, "rb") as f:
        model: Model = pickle.load(f)

    # coefs = list(enumerate(model.classifier.coef_[0]))
    # coefs.sort(key=lambda coef: coef[1])
    feature_log_prob = model.classifier.feature_log_prob_[0]

    logging.info("Reading text from stdin...")
    text = sys.stdin.read()
    (_uid, word_counts) = preprocess_message(
        {
            "uid": "0",
            "subject": "",
            "body": text,
        }
    )
    encoded = encode(model.word_encoder, word_counts)
    [pred] = model.classifier.predict([encoded])

    top_features = [
        (feature_log_prob[feature_idx], model.word_encoder[feature_idx])
        for (feature_idx, feature) in enumerate(encoded)
        if feature
    ]
    top_features.sort(key=lambda x: x[0], reverse=True)
    logging.info("Top keywords:")
    for (weight, feature) in top_features[:10]:
        logging.info("  %s: %f", feature, weight)

    [[_prob_no, prob_yes]] = model.classifier.predict_proba([encoded])
    logging.info("Prediction %d with probability %f", pred, prob_yes)


if __name__ == "__main__":
    main()
