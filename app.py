import email.utils
import os
from functools import cache
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request

from recruiterspam.classify import ClassifyResult, classify
from recruiterspam.respond import respond
from recruiterspam.train import Model, load_model

app = Flask(__name__)


@cache
def get_model() -> Model:
    return load_model(
        model_path=Path("model.pkl"),
    )


@app.route("/detect", methods=["GET", "POST"])
def detect() -> Response:
    model = get_model()
    query = request.get_json(force=True)
    assert query is not None
    subject = query.get("subject", "")
    body = query.get("body", "")
    result = classify(model, text=subject + " " + body, num_top_keywords=5)
    return jsonify(
        {
            "prediction": result.prediction,
            "probability": result.probability,
            "top_keywords": result.top_keywords,
        }
    )


@app.route("/reply", methods=["POST"])
def reply() -> Response:
    model = get_model()
    query = request.get_json(force=True)
    assert query is not None

    message_id = query["headers"].get("message_id")
    subject = query["headers"]["subject"]
    body = query["plain"]
    from_ = query["headers"].get("reply_to")
    if from_ is None:
        from_ = query["headers"]["from"]
    to = query["headers"]["to"]

    if body is None:
        print("No body available, not processing")
        return jsonify(None)

    text = subject + " " + body
    classify_result = classify(model, text=text, num_top_keywords=5)
    json_result = jsonify(
        {
            "prediction": classify_result.prediction,
            "probability": classify_result.probability,
            "top_keywords": classify_result.top_keywords,
        }
    )
    print(json_result)

    if classify_result.prediction and not subject.startswith("Re: "):
        _do_reply(
            message_id=message_id,
            subject=subject,
            text=text,
            classify_result=classify_result,
            from_=from_,
            to=to,
        )
    else:
        print(f"Skipping message: {subject}")
    return json_result


def _do_reply(
    message_id: str,
    subject: str,
    text: str,
    classify_result: ClassifyResult,
    from_: str,
    to: str,
) -> None:
    bot_domain = os.environ["EMAIL_DOMAIN"]
    bot_name = "RecruiterReplyBot"
    bot_email = f"recruiter.reply.bot@{bot_domain}"
    if bot_name in from_ or bot_name in to:
        # Try to avoid going into an infinite loop during testing.
        print(f"Skipping mail because it seems to include {bot_name} as a recipient")
        return

    reply_addresses = [from_, to]
    reply_to = to
    plain = respond(text, classify_result)
    payload = {
        "from": email.utils.formataddr((bot_name, bot_email)),
        "to": [from_, to],
        "plain": plain,
        "subject": f"Re: {subject}",
        "headers": {
            "Reply-To": to,
            "In-Reply-To": message_id,
            "References": message_id,
        },
    }

    email_api_url = os.environ["EMAIL_API_URL"]
    email_api_key = os.environ["EMAIL_API_KEY"]
    requests.post(
        email_api_url,
        headers={"Authorization": f"Bearer {email_api_key}"},
        json=payload,
    )
    print(f"Sent reply to: {reply_addresses!r}")
    print(f"Reply-To: {reply_to}")
