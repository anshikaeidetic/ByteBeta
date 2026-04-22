FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir ".[server,openai,anthropic,onnx]"

# Pre-download embedding model so cold starts don't hit HuggingFace at runtime
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('GPTCache/albert-duplicate-onnx')"

ENV BYTE_HOST=0.0.0.0

EXPOSE 8080

CMD ["sh", "-c", "byte_server --host 0.0.0.0 --port ${PORT:-8080} --gateway True --gateway-mode adaptive --gateway-cache-mode hybrid --cors-origins '*' --cache-dir /tmp/byte_cache"]
