from abc import ABCMeta, abstractmethod
from typing import Any


class ContextProcess(metaclass=ABCMeta):
    """ContextProcess: the context process interfacer, which is used to pre-process the lang conversation.
    By the way, the byte will acquire more information and get a more accurate embedding vector.

    Example:
        .. code-block:: python

            from byte.processor.context import SummarizationContextProcess

            context_process = SummarizationContextProcess()
            cache.init(pre_embedding_func=context_process.pre_process)
    """

    @abstractmethod
    def format_all_content(self, data: dict[str, Any], **params: dict[str, Any]) -> None:
        """format all content of the llm request data as a string

        :param data: the user llm request data
        :type data: Dict[str, Any]
        """

    @abstractmethod
    def process_all_content(self) -> (Any, Any):
        """process all content of the llm request data, for extracting key information in context.
        In order to achieve this goal, you can pass the summary model and so on
        """

    def pre_process(self, data: dict[str, Any], **params: dict[str, Any]) -> (Any, Any):
        """pre-process function, it's used as the byte initialization param -- pre_embedding_func.

        :param data: the user llm request data
        :type data: Dict[str, Any]
        """
        self.format_all_content(data, **params)
        return self.process_all_content()
