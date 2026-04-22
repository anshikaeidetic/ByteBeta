from typing import Any

__all__ = [
    "SBERT",
    "Cohere",
    "Data2VecAudio",
    "FastText",
    "Huggingface",
    "LangChain",
    "Onnx",
    "OpenAI",
    "PaddleNLP",
    "Rwkv",
    "Timm",
    "UForm",
    "ViT",
]


from byte.utils.lazy_import import LazyImport

openai = LazyImport("openai", globals(), "byte.embedding.openai")
huggingface = LazyImport("huggingface", globals(), "byte.embedding.huggingface")
sbert = LazyImport("sbert", globals(), "byte.embedding.sbert")
onnx = LazyImport("onnx", globals(), "byte.embedding.onnx")
cohere = LazyImport("cohere", globals(), "byte.embedding.cohere")
fasttext = LazyImport("fasttext", globals(), "byte.embedding.fasttext")
data2vec = LazyImport("data2vec", globals(), "byte.embedding.data2vec")
timm = LazyImport("timm", globals(), "byte.embedding.timm")
vit = LazyImport("vit", globals(), "byte.embedding.vit")
langchain = LazyImport("langchain", globals(), "byte.embedding.langchain")
rwkv = LazyImport("rwkv", globals(), "byte.embedding.rwkv")
paddlenlp = LazyImport("paddlenlp", globals(), "byte.embedding.paddlenlp")
uform = LazyImport("uform", globals(), "byte.embedding.uform")


def Cohere(model="large", api_key=None) -> Any:
    return cohere.Cohere(model, api_key)


def OpenAI(model="text-embedding-ada-002", api_key=None) -> Any:
    return openai.OpenAI(model, api_key)


def Huggingface(model="distilbert-base-uncased") -> Any:
    return huggingface.Huggingface(model)


def SBERT(model="all-MiniLM-L6-v2") -> Any:
    return sbert.SBERT(model)


def Onnx(model="sentence-transformers/paraphrase-albert-small-v2") -> Any:
    return onnx.Onnx(model)


def FastText(model="en", dim=None) -> Any:
    return fasttext.FastText(model, dim)


def Data2VecAudio(model="facebook/data2vec-audio-base-960h") -> Any:
    return data2vec.Data2VecAudio(model)


def Timm(model="resnet50", device="default") -> Any:
    return timm.Timm(model, device)


def ViT(model="google/vit-base-patch16-384") -> Any:
    return vit.ViT(model)


def LangChain(embeddings, dimension=0) -> Any:
    return langchain.LangChain(embeddings, dimension)


def Rwkv(model="sgugger/rwkv-430M-pile") -> Any:
    return rwkv.Rwkv(model)


def PaddleNLP(model="ernie-3.0-medium-zh") -> Any:
    return paddlenlp.PaddleNLP(model)


def UForm(model="unum-cloud/uform-vl-multilingual", embedding_type="text") -> Any:
    return uform.UForm(model, embedding_type)
