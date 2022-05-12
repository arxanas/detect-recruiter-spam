FROM python:3.9
WORKDIR /usr/src/app

ENV NLTK_DATA=./nltk_data
ENV MODEL_PATH=./model.pkl
RUN mkdir ${NLTK_DATA}

# Cache `pip install` as per https://stackoverflow.com/a/25307587
COPY requirements.txt .
COPY requirements-dev.txt .
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt
RUN python -m nltk.downloader wordnet

COPY . .

RUN env PYTHONPATH=. pytest -l -vv
RUN python -m recruiterspam.train --messages messages.json --output ${MODEL_PATH}
RUN echo "Amazing opportunities for software engineers like you" | python -m recruiterspam.classify --model ${MODEL_PATH}

CMD ["gunicorn", "app:app"]
