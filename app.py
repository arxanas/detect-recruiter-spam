from functools import cache
from pathlib import Path
from recruiterspam.classify import classify

from flask import Flask, Response, jsonify, request

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
