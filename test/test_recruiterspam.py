from typing import Any, Dict, List

import pytest
from recruiterspam.train import (
    MessageUid,
    Model,
    RawJsonMessage,
    RawJsonMessageCategory,
    count_words,
    encode,
    preprocess_message,
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
    all_word_counts = count_words(example_messages["all_messages"])
    flagged_messages = {
        MessageUid(message["uid"]) for message in example_messages["spam_messages"]
    }

    return train(all_word_counts, flagged_messages, shuffle_training_set=False)


def test_predict(example_model: Model) -> None:
    (_uid, word_counts) = preprocess_message(
        {
            "uid": "10",
            "subject": "Find a new job opportunity",
            "body": "This is an example message about increasing your salary.",
        }
    )
    encoded = encode(example_model.word_encoder, word_counts)
    assert example_model.classifier.predict([encoded])

    (_uid, word_counts) = preprocess_message(
        {
            "uid": "11",
            "subject": "I love Python",
            "body": "This is an example message about machine learning.",
        }
    )
    encoded = encode(example_model.word_encoder, word_counts)
    assert not example_model.classifier.predict([encoded])
