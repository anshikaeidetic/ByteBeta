__all__ = [
    "_check_library",
    "_missing_library_error",
    "import_boto3",
    "import_chromadb",
    "import_cohere",
    "import_diffusers",
    "import_docarray",
    "import_faiss",
    "import_fastapi",
    "import_fasttext",
    "import_hnswlib",
    "import_httpx",
    "import_huggingface",
    "import_huggingface_hub",
    "import_langchain",
    "import_llama_cpp_python",
    "import_milvus_lite",
    "import_mongodb",
    "import_onnxruntime",
    "import_openai",
    "import_paddle",
    "import_paddlenlp",
    "import_pillow",
    "import_pydantic",
    "import_pymilvus",
    "import_qdrant",
    "import_redis",
    "import_replicate",
    "import_ruamel",
    "import_sbert",
    "import_scipy",
    "import_selective_context",
    "import_sql_client",
    "import_sqlalchemy",
    "import_stability",
    "import_tiktoken",
    "import_timm",
    "import_torch",
    "import_torchaudio",
    "import_torchvision",
    "import_uform",
    "import_usearch",
    "import_vit",
    "import_weaviate",
    "lazy_optional_attr",
    "lazy_optional_module",
    "load_optional_attr",
    "load_optional_module",
    "softmax",
]

from byte.utils._optional_imports import (
    _check_library,
    _missing_library_error,
    lazy_optional_attr,
    lazy_optional_module,
    load_optional_attr,
    load_optional_module,
)
from byte.utils.softmax import softmax  # pylint: disable=unused-argument


def import_pymilvus() -> None:
    _check_library("pymilvus")


def import_milvus_lite() -> None:
    _check_library("milvus")


def import_qdrant() -> None:
    _check_library("qdrant_client")


def import_sbert() -> None:
    _check_library("sentence_transformers", package="sentence-transformers")


def import_cohere() -> None:
    _check_library("cohere")


def import_fasttext() -> None:
    _check_library("fasttext", package="fasttext==0.9.2")


def import_huggingface() -> None:
    _check_library("transformers")


def import_uform() -> None:
    _check_library("uform", package="uform==0.2.1")


def import_usearch() -> None:
    _check_library("usearch", package="usearch==0.22.3")


def import_torch() -> None:
    _check_library("torch")


def import_huggingface_hub() -> None:
    _check_library("huggingface_hub", package="huggingface-hub")


def import_onnxruntime() -> None:
    _check_library("onnxruntime", package="onnxruntime")


def import_faiss() -> None:
    _check_library("faiss", package="faiss-cpu")


def import_hnswlib() -> None:
    _check_library("hnswlib")


def import_chromadb() -> None:
    _check_library("chromadb", package="chromadb==0.3.26")


def import_sqlalchemy() -> None:
    _check_library("sqlalchemy")


def import_postgresql() -> None:
    _check_library("psycopg2", package="psycopg2-binary")


def import_pymysql() -> None:
    _check_library("pymysql")


# `brew install unixodbc` in mac
# and install PyODBC driver.
def import_pyodbc() -> None:
    _check_library("pyodbc")


# install cx-Oracle driver.
def import_cxoracle() -> None:
    _check_library("cx_Oracle")


def import_duckdb() -> None:
    _check_library("duckdb", package="duckdb")
    _check_library("duckdb-engine", package="duckdb-engine")


def import_sql_client(db_name) -> None:
    if db_name == "postgresql":
        import_postgresql()
    elif db_name in ["mysql", "mariadb"]:
        import_pymysql()
    elif db_name == "sqlserver":
        import_pyodbc()
    elif db_name == "oracle":
        import_cxoracle()
    elif db_name == "duckdb":
        import_duckdb()


def import_mongodb() -> None:
    _check_library("pymongo")
    _check_library("mongoengine")


def import_pydantic() -> None:
    _check_library("pydantic")


def import_langchain() -> None:
    _check_library("langchain")


def import_pillow() -> None:
    _check_library("PIL", package="pillow")


def import_boto3() -> None:
    _check_library("boto3")


def import_diffusers() -> None:
    _check_library("diffusers")


def import_torchaudio() -> None:
    _check_library("torchaudio")


def import_torchvision() -> None:
    _check_library("torchvision")


def import_timm() -> None:
    _check_library("timm", package="timm")


def import_vit() -> None:
    _check_library("vit", package="vit")


def import_replicate() -> None:
    _check_library("replicate")


def import_stability() -> None:
    _check_library("stability_sdk", package="stability-sdk")


def import_scipy() -> None:
    _check_library("scipy")


def import_llama_cpp_python() -> None:
    _check_library("llama_cpp", package="llama-cpp-python")


def import_ruamel() -> None:
    _check_library("ruamel.yaml", package="ruamel-yaml")


def import_selective_context() -> None:
    _check_library("selective_context")


def import_httpx() -> None:
    _check_library("httpx")


def import_openai() -> None:
    _check_library("openai", package="openai")


def import_anthropic() -> None:
    _check_library("anthropic")


def import_google_genai() -> None:
    _check_library("google.genai", package="google-genai")


def import_ollama() -> None:
    _check_library("ollama")


def import_docarray() -> None:
    _check_library("docarray")


def import_paddle() -> None:
    _check_library("google.protobuf", package="protobuf==3.20.0")
    _check_library("paddle", package="paddlepaddle")


def import_paddlenlp() -> None:
    _check_library("paddlenlp")


def import_tiktoken() -> None:
    _check_library("tiktoken")


def import_fastapi() -> None:
    _check_library("fastapi")


def import_redis() -> None:
    _check_library("redis")
    _check_library("redis_om", package="redis-om")


def import_starlette() -> None:
    _check_library("starlette")


def import_weaviate() -> None:
    _check_library("weaviate", package="weaviate-client")
