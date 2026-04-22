FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir ".[server,openai,anthropic,onnx,sql]"

# Pre-bake the ONNX model and its tokenizer into the image
RUN HF_HOME=/app/.hf_cache python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('GPTCache/albert-duplicate-onnx'); \
snapshot_download('sentence-transformers/paraphrase-albert-small-v2')"

# At runtime: use baked models, block all HuggingFace network calls
ENV HF_HOME=/app/.hf_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV BYTE_HOST=0.0.0.0
ENV BYTE_DEMO_PATH=/app/demo.html

EXPOSE 8080

CMD ["sh", "-c", "byte_server --host 0.0.0.0 --port ${PORT:-8080} --gateway True --gateway-mode adaptive --gateway-cache-mode hybrid --cors-origins '*' --cache-dir /tmp/byte_cache"]
