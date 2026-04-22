from collections.abc import Mapping
from typing import Any

from byte.adapter.adapter import aadapt, adapt
from byte.core import cache
from byte.manager.scalar_data.base import Answer, DataType
from byte.session import Session

try:
    from langchain_core.language_models.llms import BaseLLM as LLM
except ImportError:
    try:
        from langchain.llms.base import LLM
    except ImportError:
        from byte.utils import import_langchain

        import_langchain()
        from langchain.llms.base import LLM

try:
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:
    try:
        from langchain.chat_models.base import BaseChatModel
    except ImportError:
        from byte.utils import import_langchain

        import_langchain()
        from langchain.chat_models.base import BaseChatModel

try:
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult, Generation, LLMResult
except ImportError:
    from langchain.schema import (
        AIMessage,
        BaseMessage,
        ChatGeneration,
        ChatResult,
        Generation,
        LLMResult,
    )

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
        Callbacks,
    )
except ImportError:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
        Callbacks,
    )


# pylint: disable=protected-access
class LangChainLLMs(LLM):
    """LangChain LLM Wrapper.

    :param llm: LLM from langchain.llms.
    :type llm: Any

    Example:
        .. code-block:: python

            from byte import cache
            from byte.processor.pre import get_prompt
            # init byte
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()

            from langchain.llms import OpenAI
            from byte.adapter.langchain_models import LangChainLLMs
            # run llm with byte
            llm = LangChainLLMs(llm=OpenAI(temperature=0))
            llm("Hello world")
    """

    llm: Any
    cache_obj: Any = cache
    session: Session = None
    tmp_args: Any = None

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self.llm._identifying_params

    def __str__(self) -> str:
        return str(self.llm)

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        _: CallbackManagerForLLMRun | None = None,
    ) -> str:
        self.tmp_args = dict(self.tmp_args or {})
        session = self.session if "session" not in self.tmp_args else self.tmp_args.pop("session")
        cache_obj = self.tmp_args.pop("cache_obj", self.cache_obj)

        def llm_handler(*_, **llm_kwargs) -> Any:
            llm_kwargs.pop("prompt", None)
            llm_kwargs.pop("cache_obj", None)
            llm_kwargs.pop("session", None)
            llm_kwargs.pop("stop", None)
            return _invoke_langchain_llm(self.llm, prompt, stop=stop, **llm_kwargs)

        return adapt(
            llm_handler,
            _cache_data_convert,
            _update_cache_callback,
            prompt=prompt,
            stop=stop,
            cache_obj=cache_obj,
            session=session,
            **self.tmp_args,
        )

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, _=run_manager)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations, llm_output=None)

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
    ) -> str:
        self.tmp_args = dict(self.tmp_args or {})
        session = self.session if "session" not in self.tmp_args else self.tmp_args.pop("session")
        cache_obj = self.tmp_args.pop("cache_obj", self.cache_obj)

        async def llm_handler(*_, **llm_kwargs) -> Any:
            llm_kwargs.pop("prompt", None)
            llm_kwargs.pop("cache_obj", None)
            llm_kwargs.pop("session", None)
            llm_kwargs.pop("stop", None)
            return await _ainvoke_langchain_llm(self.llm, prompt, stop=stop, **llm_kwargs)

        return await aadapt(
            llm_handler,
            _cache_data_convert,
            _update_cache_callback,
            prompt=prompt,
            stop=stop,
            cache_obj=cache_obj,
            session=session,
            **self.tmp_args,
        )

    def generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        callbacks: Callbacks | None = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return super().generate(prompts, stop=stop, callbacks=callbacks)

    async def agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        callbacks: Callbacks | None = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return await super().agenerate(prompts, stop=stop, callbacks=callbacks)

    def __call__(
        self,
        prompt: str,
        stop: list[str] | None = None,
        callbacks: Callbacks | None = None,
        **kwargs,
    ) -> str:
        """Check Cache and run the LLM on the given prompt and input."""
        return (
            self.generate([prompt], stop=stop, callbacks=callbacks, **kwargs).generations[0][0].text
        )


# pylint: disable=protected-access
class LangChainChat(BaseChatModel):
    """LangChain Chat Model Wrapper.

    :param chat: LLM from langchain.chat_models.
    :type chat: Any

    Example:
        .. code-block:: python

            from byte import cache
            from byte.processor.pre import get_messages_last_content
            # init byte
            cache.init(pre_embedding_func=get_messages_last_content)
            cache.set_openai_key()
            from langchain.chat_models import ChatOpenAI
            from byte.adapter.langchain_models import LangChainChat
            # run chat model with byte
            chat = LangChainChat(chat=ChatOpenAI(temperature=0))
            chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
    """

    @property
    def _llm_type(self) -> str:
        return "byte_llm_chat"

    chat: Any
    cache_obj: Any = cache
    session: Session | None = None
    tmp_args: Any | None = None

    def _generate(
        self,
        messages: Any,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
    ) -> ChatResult:
        self.tmp_args = dict(self.tmp_args or {})
        cache_messages = _normalize_langchain_messages(messages)
        session = self.session if "session" not in self.tmp_args else self.tmp_args.pop("session")
        cache_obj = self.tmp_args.pop("cache_obj", self.cache_obj)

        def llm_handler(*_, **llm_kwargs) -> Any:
            llm_kwargs.pop("messages", None)
            llm_kwargs.pop("cache_obj", None)
            llm_kwargs.pop("session", None)
            llm_kwargs.pop("stop", None)
            llm_kwargs.pop("run_manager", None)
            return _invoke_langchain_chat(
                self.chat,
                messages,
                stop=stop,
                **llm_kwargs,
            )

        return adapt(
            llm_handler,
            _cache_msg_data_convert,
            _update_cache_msg_callback,
            messages=cache_messages,
            stop=stop,
            cache_obj=cache_obj,
            session=session,
            run_manager=run_manager,
            **self.tmp_args,
        )

    async def _agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
    ) -> ChatResult:
        self.tmp_args = dict(self.tmp_args or {})
        cache_messages = _normalize_langchain_messages(messages)
        session = self.session if "session" not in self.tmp_args else self.tmp_args.pop("session")
        cache_obj = self.tmp_args.pop("cache_obj", self.cache_obj)

        async def llm_handler(*_, **llm_kwargs) -> Any:
            llm_kwargs.pop("messages", None)
            llm_kwargs.pop("cache_obj", None)
            llm_kwargs.pop("session", None)
            llm_kwargs.pop("stop", None)
            llm_kwargs.pop("run_manager", None)
            return await _ainvoke_langchain_chat(
                self.chat,
                messages,
                stop=stop,
                **llm_kwargs,
            )

        return await aadapt(
            llm_handler,
            _cache_msg_data_convert,
            _update_cache_msg_callback,
            messages=cache_messages,
            stop=stop,
            cache_obj=cache_obj,
            session=session,
            run_manager=run_manager,
            **self.tmp_args,
        )

    def generate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks | None = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return super().generate(messages, stop=stop, callbacks=callbacks)

    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks | None = None,
        **kwargs,
    ) -> LLMResult:
        self.tmp_args = kwargs
        return await super().agenerate(messages, stop=stop, callbacks=callbacks)

    @property
    def _identifying_params(self) -> Any:
        return self.chat._identifying_params

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:
        return self.chat._combine_llm_outputs(llm_outputs)

    def get_num_tokens(self, text: str) -> int:
        return self.chat.get_num_tokens(text)

    def get_num_tokens_from_messages(self, messages: list[BaseMessage]) -> int:
        return self.chat.get_num_tokens_from_messages(messages)

    def __call__(self, messages: Any, stop: list[str] | None = None, **kwargs) -> Any:
        generation = self.generate([messages], stop=stop, **kwargs).generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")


def _cache_data_convert(cache_data) -> Any:
    return cache_data


def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
    update_cache_func(Answer(llm_data, DataType.STR))
    return llm_data


def _cache_msg_data_convert(cache_data) -> Any:
    llm_res = ChatResult(
        generations=[
            ChatGeneration(
                text="",
                generation_info=None,
                message=AIMessage(content=cache_data, additional_kwargs={}),
            )
        ],
        llm_output=None,
    )
    return llm_res


def _update_cache_msg_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
    update_cache_func(llm_data.generations[0].text)
    return llm_data


def _normalize_langchain_messages(messages: Any) -> Any:
    if isinstance(messages, BaseMessage):
        return _message_to_payload(messages)
    if isinstance(messages, list):
        return [_normalize_langchain_messages(message) for message in messages]
    return messages


def _message_to_payload(message: Any) -> Any:
    if isinstance(message, BaseMessage):
        role = getattr(message, "type", "")
        role = {
            "human": "user",
            "ai": "assistant",
        }.get(role, role)
        return {
            "role": role,
            "content": getattr(message, "content", ""),
        }
    return message


def _response_mapping(response: Any) -> Mapping[str, Any] | None:
    if isinstance(response, Mapping):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    as_dict = getattr(response, "dict", None)
    if callable(as_dict):
        dumped = as_dict()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _extract_text_response(response: Any) -> str:
    mapping = _response_mapping(response)
    if mapping is not None:
        choices = mapping.get("choices") or []
        if choices:
            first_choice = choices[0] or {}
            if isinstance(first_choice, Mapping):
                text = first_choice.get("text")
                if isinstance(text, str):
                    return text
                message = first_choice.get("message") or {}
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
        text = mapping.get("text")
        if isinstance(text, str):
            return text
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(response, str):
        return response
    raise TypeError(f"Unsupported LangChain completion response type: {type(response)!r}")


def _extract_chat_content(response: Any) -> str:
    mapping = _response_mapping(response)
    if mapping is not None:
        choices = mapping.get("choices") or []
        if choices:
            first_choice = choices[0] or {}
            if isinstance(first_choice, Mapping):
                message = first_choice.get("message") or {}
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                text = first_choice.get("text")
                if isinstance(text, str):
                    return text
        content = mapping.get("content")
        if isinstance(content, str):
            return content
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(response, BaseMessage):
        return getattr(response, "content", "")
    raise TypeError(f"Unsupported LangChain chat response type: {type(response)!r}")


def _chat_result_from_response(response: Any) -> ChatResult:
    content = _extract_chat_content(response)
    return ChatResult(
        generations=[
            ChatGeneration(
                text=content,
                generation_info=None,
                message=AIMessage(content=content, additional_kwargs={}),
            )
        ],
        llm_output=_response_mapping(response),
    )


def _invoke_langchain_llm(llm: Any, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
    payload = dict(getattr(llm, "_invocation_params", {}))
    payload.update(kwargs)
    payload["prompt"] = prompt
    if stop is not None:
        payload["stop"] = stop
    client = getattr(llm, "client", None)
    if client is not None and hasattr(client, "create"):
        return _extract_text_response(client.create(**payload))
    if hasattr(llm, "invoke"):
        return _extract_text_response(llm.invoke(prompt, stop=stop, **kwargs))
    if callable(llm):
        return _extract_text_response(llm(prompt=prompt, stop=stop, **kwargs))
    raise TypeError(f"Unsupported LangChain LLM type: {type(llm)!r}")


async def _ainvoke_langchain_llm(
    llm: Any, prompt: str, stop: list[str] | None = None, **kwargs
) -> str:
    payload = dict(getattr(llm, "_invocation_params", {}))
    payload.update(kwargs)
    payload["prompt"] = prompt
    if stop is not None:
        payload["stop"] = stop
    async_client = getattr(llm, "async_client", None)
    if async_client is not None and hasattr(async_client, "create"):
        return _extract_text_response(await async_client.create(**payload))
    if hasattr(llm, "ainvoke"):
        return _extract_text_response(await llm.ainvoke(prompt, stop=stop, **kwargs))
    return _invoke_langchain_llm(llm, prompt, stop=stop, **kwargs)


def _invoke_langchain_chat(
    chat: Any,
    messages: Any,
    stop: list[str] | None = None,
    **kwargs,
) -> ChatResult:
    payload_builder = getattr(chat, "_get_request_payload", None)
    if callable(payload_builder):
        payload = payload_builder(messages, stop=stop, **kwargs)
    else:
        payload = dict(getattr(chat, "_default_params", {}))
        payload.update(kwargs)
        payload["messages"] = _normalize_langchain_messages(messages)
        if stop is not None:
            payload["stop"] = stop
    client = getattr(chat, "client", None)
    if client is not None and hasattr(client, "create"):
        return _chat_result_from_response(client.create(**payload))
    if hasattr(chat, "invoke"):
        return _chat_result_from_response(chat.invoke(messages, stop=stop, **kwargs))
    raise TypeError(f"Unsupported LangChain chat model type: {type(chat)!r}")


async def _ainvoke_langchain_chat(
    chat: Any,
    messages: Any,
    stop: list[str] | None = None,
    **kwargs,
) -> ChatResult:
    payload_builder = getattr(chat, "_get_request_payload", None)
    if callable(payload_builder):
        payload = payload_builder(messages, stop=stop, **kwargs)
    else:
        payload = dict(getattr(chat, "_default_params", {}))
        payload.update(kwargs)
        payload["messages"] = _normalize_langchain_messages(messages)
        if stop is not None:
            payload["stop"] = stop
    async_client = getattr(chat, "async_client", None)
    if async_client is not None and hasattr(async_client, "create"):
        return _chat_result_from_response(await async_client.create(**payload))
    if hasattr(chat, "ainvoke"):
        return _chat_result_from_response(await chat.ainvoke(messages, stop=stop, **kwargs))
    return _invoke_langchain_chat(chat, messages, stop=stop, **kwargs)
