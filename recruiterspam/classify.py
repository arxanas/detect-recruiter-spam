import argparse
from dataclasses import dataclass
import logging
import re
import sys
from pathlib import Path
from typing import List

from .train import load_model, Model


@dataclass(eq=True, frozen=True)
class ClassifyResult:
    prediction: bool
    probability: float
    top_keywords: List[str]


PICTURE_RE = re.compile(
    r"""
    (
        \[
            https?://
            [^]]*
        \]
        |
        \[
            cid:
            [^\]]*
        \]
    )
""",
    re.VERBOSE,
)


def remove_lines_after_picture(text: str) -> str:
    """Search the body and remove any lines including and after a picture.

    This is because some email signatures include images. In particular, they
    oftentimes contain extra text afterward with a legal disclaimer.
    """
    lines = []
    for line in text.splitlines():
        if PICTURE_RE.search(line) is not None:
            break
        lines.append(line)
    return "".join(line + "\n" for line in lines)


def classify(model: Model, text: str, num_top_keywords: int) -> ClassifyResult:
    text = remove_lines_after_picture(text)
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
        prediction=bool(prob_yes > 0.95),
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
