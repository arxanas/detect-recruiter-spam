from recruiterspam.classify import classify
from recruiterspam.backtest import backtest
from typing import Any, Dict, List
import numpy as np

import pytest
from sklearn.feature_extraction.text import CountVectorizer
from recruiterspam.train import (
    MODEL_VERSION,
    Model,
    RawJsonMessage,
    RawJsonMessageCategory,
    init_tokenizer,
    preprocess_message,
    tokenize,
    train,
)


@pytest.fixture
def example_messages() -> Dict[RawJsonMessageCategory, List[RawJsonMessage]]:
    all_messages: List[RawJsonMessage] = [
        {
            "uid": "1",
            "subject": "This is some regular old message",
            "body": "Hello, friend, I am writing a message to you.",
        },
        {
            "uid": "2",
            "subject": "Machine learning is great",
            "body": "Surely a machine learning model would not flag this message!",
        },
        {
            "uid": "3",
            "subject": "Opportunities at Company1",
            "body": "Company1 is seeking experienced engineers who have preferably not set up an anti-recruiting email filter.",
        },
        {
            "uid": "4",
            "subject": "Want to increase your compensation?",
            "body": "Join Company2 for a sure chance to increase your compensation, despite me not knowing anything about your current salary.",
        },
        {
            "uid": "5",
            "subject": "Python newletter",
            "body": "Learn more about sklearn so that you can continue writing very simple Bayesian spam filters.",
        },
        {
            "uid": "6",
            "subject": "Get more money or something",
            "body": "Opportunities available at Company2",
        },
        {
            "uid": "7",
            "subject": "Python is a cool programming language",
            "body": "But have you considered using Rust instead?",
        },
        {
            "uid": "8",
            "subject": "Can you whip up buttery-smooth code?",
            "body": "If so, contact my faceless recruiting firm.",
        },
    ]
    spam_message_uids = {"3", "4", "6", "8"}
    return {
        "all_messages": all_messages,
        "spam_messages": [m for m in all_messages if m["uid"] in spam_message_uids],
    }


@pytest.fixture
def example_model(
    example_messages: Dict[RawJsonMessageCategory, List[RawJsonMessage]]
) -> Model:
    init_tokenizer()
    count_vectorizer = CountVectorizer(
        preprocessor=preprocess_message,
        tokenizer=tokenize,
    )
    X = count_vectorizer.fit_transform(example_messages["all_messages"])
    flagged_messages = {m["uid"] for m in example_messages["spam_messages"]}
    y = np.array(
        [int(m["uid"] in flagged_messages) for m in example_messages["all_messages"]]
    )

    classifier = train(X, y, shuffle_training_set=False)
    return Model(
        model_version=MODEL_VERSION,
        count_vectorizer=count_vectorizer,
        classifier=classifier,
    )


def test_predict(example_model: Model) -> None:
    X = example_model.count_vectorizer.transform(
        [
            {
                "uid": "10",
                "subject": "Find a new job opportunity",
                "body": "This is an example message about increasing your salary.",
            }
        ]
    )
    y_pred = example_model.classifier.predict(X)
    assert y_pred == [1]

    X = example_model.count_vectorizer.transform(
        [
            {
                "uid": "11",
                "subject": "I love Python",
                "body": "This is an example message about machine learning.",
            }
        ]
    )
    y_pred = example_model.classifier.predict(X)
    assert y_pred == [0]


def test_backtest(example_messages, example_model) -> None:
    (true_positives, false_positives) = backtest(
        model=example_model, messages=example_messages
    )
    assert true_positives == 3
    assert false_positives == 0


def test_classify(example_model) -> None:
    result = classify(
        model=example_model,
        text="Get more compensation by getting a job with Company2",
        num_top_keywords=5,
    )
    assert result.prediction == True
    assert result.probability >= 0.9
    assert result.top_keywords == ["company2", "compensation", "get", "more"]

    result = classify(
        model=example_model,
        text="Python vs. Rust: who would win?",
        num_top_keywords=5,
    )
    assert result.prediction == False
    assert result.probability <= 0.3
    assert result.top_keywords == ["python", "rust", "would", "who"]
