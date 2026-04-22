FROM python:3.8-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1

RUN pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir ".[server]"

CMD ["byte_server", "-s", "0.0.0.0"]
