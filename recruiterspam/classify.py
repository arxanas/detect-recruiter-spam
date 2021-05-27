import argparse
from dataclasses import dataclass
import logging
import sys
from pathlib import Path
from typing import List

from .train import load_model, Model


@dataclass(eq=True, frozen=True)
class ClassifyResult:
    prediction: bool
    probability: float
    top_keywords: List[str]


def classify(model: Model, text: str, num_top_keywords: int) -> ClassifyResult:
    X = model.count_vectorizer.transform(
        [
            {
                "uid": "0",
                "subject": "",
                "body": text,
            }
        ]
    )
    [pred] = model.classifier.predict(X)
    [[_prob_no, prob_yes]] = model.classifier.predict_proba(X)
    logging.info("Prediction %s with probability %f", bool(pred), prob_yes)

    [X0] = X.toarray()
    feature_names = model.count_vectorizer.get_feature_names()
    words = [
        (
            feature_names[word_idx],
            model.classifier.feature_log_prob_[0, word_idx],
            model.classifier.feature_log_prob_[1, word_idx],
        )
        for word_idx in X0.nonzero()[0]
        if 0 <= word_idx < len(feature_names)
    ]
    words.sort(key=lambda x: min(x[1], x[2]))
    words = words[:num_top_keywords]
    logging.info("Top keywords (word, non-spam probability, spam probability)")
    for (word, negative_prob, positive_prob) in words:
        logging.info("  %s\t%f\t%f", word, negative_prob, positive_prob)

    return ClassifyResult(
        # Don't use `numpy` numeric values, to ensure that they can be serialized.
        prediction=bool(pred),
        probability=float(prob_yes),
        top_keywords=[word for (word, _neg_prob, _pos_prob) in words],
    )


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

    model = load_model(model_path)

    logging.info("Reading text from stdin...")
    text = sys.stdin.read()
    classify(model=model, text=text, num_top_keywords=5)


if __name__ == "__main__":
    main()
