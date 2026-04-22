"""Request accessor helpers extracted from ``byte.processor.pre``."""

import re
import string
from typing import Any

from byte.processor._pre_canonicalize import canonicalize_text
from byte.processor._pre_context_aux import _fallback_request_signature
from byte.processor._pre_selection import _message_content
from byte.utils.multimodal import content_signature


def last_content(data: dict[str, Any], **_: dict[str, Any]) -> Any:
    """get the last content of the message list

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import last_content

            content = last_content({"messages": [{"content": "foo1"}, {"content": "foo2"}]})
            # content = "foo2"
    """
    messages = data.get("messages")
    if messages:
        content = _message_content(messages[-1])
        if isinstance(content, list):
            return content_signature(content)
        return content
    return _fallback_request_signature(data)


def normalized_last_content(data: dict[str, Any], **_: dict[str, Any]) -> str:
    """Return the last message content after lightweight normalization."""
    return canonicalize_text(last_content(data))


def last_content_without_prompt(data: dict[str, Any], **params: dict[str, Any]) -> Any:
    """get the last content of the message list without prompts content

    :param data: the user llm request data
    :type data: Dict[str, Any]
    :param params: the special byte params, like prompts param in the cache object
    :type params: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import last_content_without_prompt

            content = last_content_without_prompt(
                    {"messages": [{"content": "foo1"}, {"content": "foo2"}]}, prompts=["foo"]
                )
            # content = "2"
    """

    last_content_str = data.get("messages")[-1]["content"]
    prompts = params.get("prompts", [])
    if prompts is None:
        return last_content_str
    pattern = "|".join(prompts)
    new_content_str = re.sub(pattern, "", last_content_str)
    return new_content_str


def _get_pattern_value(pattern_str: str, value_str: str) -> Any:
    literal_text_arr = []
    field_name_arr = []
    for literal_text, field_name, _, _ in string.Formatter().parse(pattern_str):
        literal_text_arr.append(literal_text)
        if field_name is not None:
            field_name_arr.append(field_name if field_name else str(len(field_name_arr)))

    pattern_values = {}
    last_end = 0
    for i, literal_text in enumerate(literal_text_arr):
        start = value_str.find(literal_text, last_end)
        if i == len(literal_text_arr) - 1:
            end = len(value_str)
        else:
            end = value_str.find(literal_text_arr[i + 1], start + 1)
        if start == -1 or end == -1:
            break
        start += len(literal_text)
        pattern_values[field_name_arr[i]] = value_str[start:end]
        last_end = end
    return pattern_values


def last_content_without_template(data: dict[str, Any], **params: dict[str, Any]) -> Any:
    """get the last content's template values of the message list without template content.

    When considering a cache agent or chain, the majority of the content consists of template content,
    while the essential information is simply a list of parameters within the template.
    In this way, the cache key is composed of a string made up of all the parameter values in the list.

    WARNING: Two parameters without intervals cannot appear in the template,
    for example: template = "{foo}{hoo}" is not supported,
    but template = "{foo}:{hoo}" is supported

    :param data: the user llm request data
    :type data: Dict[str, Any]

    :Example with str template:
        .. code-block:: python

            from byte import Config
            from byte.processor.pre import last_content_without_template

            template_obj = "tell me a joke about {subject}"
            prompt = template_obj.format(subject="animal")
            value = last_content_without_template(
                data={"messages": [{"content": prompt}]}, cache_config=Config(template=template_obj)
            )
            print(value)
            # ['animal']

    :Example with langchain template:
        .. code-block:: python

            from langchain import PromptTemplate

            from byte import Config
            from byte.processor.pre import last_content_without_template

            template_obj = PromptTemplate.from_template("tell me a joke about {subject}")
            prompt = template_obj.format(subject="animal")

            value = last_content_without_template(
                data={"messages": [{"content": prompt}]},
                cache_config=Config(template=template_obj.template),
            )
            print(value)
            # ['animal']

    NOTE: At present, only the simple PromptTemplate in langchain is supported.
    For ChatPromptTemplate, it needs to be adjusted according to the template array.
    If you need to use it, you need to pass in the final dialog template yourself.
    The reason why it cannot be advanced is that ChatPromptTemplate
    does not provide a method to directly return the template string.
    """
    last_content_str = data.get("messages")[-1]["content"]
    cache_config = params.get("cache_config")
    if not (cache_config and cache_config.template):
        return last_content_str

    pattern_value = _get_pattern_value(cache_config.template, last_content_str)
    return str(list(pattern_value.values()))


def all_content(data: dict[str, Any], **_: dict[str, Any]) -> Any:
    """get all content of the message list

    :param data: the user llm request data
    :type data: Dict[str, Any]

    :Example:
        .. code-block:: python

            from byte.processor.pre import all_content

            content = all_content(
                {"messages": [{"content": "foo1"}, {"content": "foo2"}]}
            )
            # content = "foo1\\nfoo2"
    """
    s = ""
    messages = data.get("messages")
    for i, message in enumerate(messages):
        content = message["content"]
        if isinstance(content, list):
            content = content_signature(content)
        if i == len(messages) - 1:
            s += content
        else:
            s += content + "\n"
    return s


def nop(data: dict[str, Any], **_: dict[str, Any]) -> Any:
    """do nothing of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import nop

            content = nop({"str": "hello"})
            # {"str": "hello"}
    """
    return data


def get_prompt(data: dict[str, Any], **_: dict[str, Any]) -> Any:
    """get the prompt of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_prompt

            content = get_prompt({"prompt": "foo"})
            # "foo"
    """
    return data.get("prompt")


def normalized_get_prompt(data: dict[str, Any], **_: dict[str, Any]) -> str:
    """Return the prompt after lightweight normalization."""
    return canonicalize_text(get_prompt(data))


def get_file_name(data: dict[str, Any], **_: dict[str, Any]) -> str:
    """get the file name of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_file_name

            file = open("test.txt", "a")
            content = get_file_name({"file": file})
            # "test.txt"
    """
    file_obj = data.get("file")
    if isinstance(file_obj, dict):
        return str(file_obj.get("name") or "")
    return file_obj.name


def get_file_bytes(data: dict[str, Any], **_: dict[str, Any]) -> bytes:
    """get the file bytes of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_file_bytes

            content = get_file_bytes({"file": open("test.txt", "rb")})
    """
    file_obj = data.get("file")
    if isinstance(file_obj, dict):
        payload = file_obj.get("bytes", b"")
        return payload if isinstance(payload, bytes) else bytes(payload)
    if hasattr(file_obj, "read"):
        position = file_obj.tell() if hasattr(file_obj, "tell") else None
        payload = file_obj.read()
        if position is not None and hasattr(file_obj, "seek"):
            file_obj.seek(position)
        return payload if isinstance(payload, bytes) else bytes(payload)
    return file_obj.peek()


def get_input_str(data: dict[str, Any], **_: dict[str, Any]) -> str:
    """get the image and question str of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_input_str

            content = get_input_str({"input": {"image": open("test.png", "rb"), "question": "foo"}})
    """
    input_data = data.get("input")
    return str(input_data["image"].peek()) + input_data["question"]


def get_input_image_file_name(data: dict[str, Any], **_: dict[str, Any]) -> str:
    """get the image file name of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_input_image_file_name

            content = get_input_image_file_name({"input": {"image": open("test.png", "rb")}})
            # "test.png"
    """
    input_data = data.get("input")
    return input_data["image"].name


def get_image_question(data: dict[str, Any], **_: dict[str, Any]) -> str:  # pragma: no cover
    """get the image and question str of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_image_question

            content = get_image_question({"image": open("test.png", "rb"), "question": "foo"})
    """
    img = data.get("image")
    if isinstance(img, str):
        with open(img, "rb") as handle:
            data_img = str(handle.peek())
    else:
        data_img = str(img)
    return data_img + data.get("question")


def get_image(data: dict[str, Any], **_: dict[str, Any]) -> str:  # pragma: no cover
    """get the image of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_image

            content = get_image({"image": open("test.png", "rb")})
            # "test.png"
    """
    return data.get("image")


def get_inputs(data: dict[str, Any], **_: dict[str, Any]) -> Any:
    """get the inputs of the llm request params

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_inputs

            content = get_inputs({"inputs": "hello"})
            # "hello"
    """
    return data.get("inputs")


def get_messages_last_content(data: dict[str, Any], **_: Any) -> str:
    """get the last content of the llm request messages array

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_messages_last_content

            content = get_messages_last_content({"messages": [{"content": "hello"}, {"content": "world"}]})
            # "world"
    """
    messages = data.get("messages") or []
    if not messages:
        return ""
    content = _message_content(messages[-1])
    return content_signature(content) if isinstance(content, list) else str(content or "")


def get_openai_moderation_input(data: dict[str, Any], **_: dict[str, Any]) -> str:
    """get the input param of the openai moderation request params

    :param data: the user openai moderation request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import get_openai_moderation_input

            content = get_openai_moderation_input({"input": ["hello", "world"]})
            # "['hello', 'world']"
    """

    return str(data.get("input"))


def concat_all_queries(data: dict[str, Any], **params: dict[str, Any]) -> Any:
    """

    :param data: the user llm request data
    :type data: Dict[str, Any]

    Example:
        .. code-block:: python

            from byte.processor.pre import concat_all_queries

            content = concat_all_queries({"messages": [{"role": "system", "content": "hello"},
                {"role": "user", "content": "world"},
                {"role": "assistant", "content": "alice"}]})

    """
    cache_config = params.get("cache_config")
    skip_list = cache_config.skip_list
    context_len = cache_config.context_len
    context_len = context_len * 2
    s = ""
    messages = data.get("messages")
    length = min(context_len, len(messages))
    messages = messages[len(messages) - length :]
    for i, message in enumerate(messages):
        if message["role"] in skip_list:
            continue
        if i == len(messages) - 1:
            s += f"{message['role'].upper()}: {message['content']}"
        else:
            s += f"{message['role'].upper()}: {message['content']}\n"
    return s


__all__ = [name for name in globals() if not name.startswith("__")]
