import email.utils
import smtplib
import ssl
import os
from email.message import EmailMessage
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


@app.route("/health")
def health() -> Response:
    model = get_model()
    return f"Loaded model version {model.model_version}"


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
    in_reply_to = query["headers"].get("in_reply_to")

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

    if not classify_result.prediction:
        print(f"Skipping message (not recruiter-spam): {subject}")
    elif subject.startswith("Re: ") or in_reply_to is not None:
        print(f"Skipping message (appears to be a reply): {subject}")
    else:
        _do_reply(
            message_id=message_id,
            subject=subject,
            text=text,
            classify_result=classify_result,
            from_=from_,
            to=to,
        )
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
    bot_name = os.environ.get("EMAIL_NAME", "RecruiterReplyBot")
    bot_email = os.environ.get("EMAIL_ADDRESS", f"recruiter.reply.bot@{bot_domain}")

    reply_addresses = [from_, to]
    reply_to = to
    plain = respond(text, classify_result)

    bot_from_address = email.utils.formataddr((bot_name, bot_email))

    gmail_username = os.environ.get("GMAIL_USERNAME")
    if gmail_username is not None:
        message = EmailMessage()
        message.set_content(plain)
        message["From"] = bot_from_address
        message["To"] = ", ".join(reply_addresses)
        message["Reply-To"] = to
        message["In-Reply-To"] = message_id
        message["References"] = message_id
        message["Subject"] = f"Re: {subject}"

        gmail_app_password = os.environ["GMAIL_APP_PASSWORD"]
        # As per https://support.google.com/a/answer/176600?hl=en
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            host="smtp.gmail.com",
            port=465,
            context=context,
        ) as smtp:
            smtp.login(gmail_username, gmail_app_password)
            smtp.send_message(message)
            print(f"Sent reply to (via Gmail): {reply_addresses!r}")
            print(f"Reply-To: {reply_to}")

    else:
        email_api_url = os.environ["EMAIL_API_URL"]
        email_api_key = os.environ["EMAIL_API_KEY"]
        payload = {
            "from": bot_from_address,
            "to": [from_, to],
            "plain": plain,
            "subject": f"Re: {subject}",
            "headers": {
                "Reply-To": to,
                "In-Reply-To": message_id,
                "References": message_id,
            },
        }
        requests.post(
            email_api_url,
            headers={"Authorization": f"Bearer {email_api_key}"},
            json=payload,
        )
        print(f"Sent reply to (via API): {reply_addresses!r}")
        print(f"Reply-To: {reply_to}")
