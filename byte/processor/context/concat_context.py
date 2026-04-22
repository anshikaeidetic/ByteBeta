from typing import Any

from byte.processor import ContextProcess


class ConcatContextProcess(ContextProcess):
    """A concat context processor simply concat the context.
    Generally used with rwkv embedding, because rwkv can input almost infinitely long

    Example:
        .. code-block:: python

            from byte.manager import manager_factory
            from byte.processor.context.concat_context import ConcatContextProcess

            context_process = ConcatContextProcess()
            rwkv_embedding = Rwkv()
            data_manager = manager_factory(
                "sqlite,faiss",
                vector_params={"dimension": rwkv_embedding.dimension},
            )
            cache.init(
                pre_embedding_func=context_process.pre_process,
                embedding_func=rwkv_embedding.to_embeddings,
                data_manager=data_manager,
            )
    """

    content: str = ""

    def __init__(self) -> None:
        self.content = ""
        self.concat_content = ""

    def format_all_content(self, data: dict[str, Any], **params: dict[str, Any]) -> None:
        for query in data["messages"]:
            self.content += f"{query['role']}: {query['content']} \n"
            self.concat_content += query["content"]

    def process_all_content(self) -> (Any, Any):
        return self.content, self.concat_content
