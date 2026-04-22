FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir ".[server,openai,anthropic]"

RUN mkdir -p /data/byte_cache

ENV BYTE_HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "byte_server --host 0.0.0.0 --port ${PORT} --gateway True --gateway-mode adaptive --gateway-cache-mode hybrid --cors-origins '*' --cache-dir /data/byte_cache"]
