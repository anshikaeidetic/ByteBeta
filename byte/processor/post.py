import random
from typing import Any

import numpy

from byte.utils import softmax
from byte.utils.log import byte_log


def _resolve_completion_create(client) -> Any:
    if client is None:
        import openai  # pylint: disable=C0415

        client = openai

    chat = getattr(client, "chat", None)
    chat_completions = getattr(chat, "completions", None)
    if hasattr(chat_completions, "create"):
        return chat_completions.create

    completions = getattr(client, "completions", None)
    if hasattr(completions, "create"):
        return completions.create

    legacy_chat = getattr(client, "ChatCompletion", None)
    if hasattr(legacy_chat, "create"):
        return legacy_chat.create

    raise AttributeError("Client does not expose a supported completion create method")


def _extract_completion_text(response) -> str:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return ""
        first_choice = choices[0]
        message = first_choice.get("message") or {}
        return str(message.get("content") or first_choice.get("text") or "")

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)

    text = getattr(first_choice, "text", None)
    return str(text or "")


def random_one(messages: list[Any]) -> Any:
    """Randomly select one result after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from byte.processor.post import random_one

            messages = ["message 1", "message 2", "message 3"]
            answer = random_one(messages)
    """
    return random.choice(messages)


def first(messages: list[Any]) -> Any:
    """Get the first result after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from byte.processor.post import first

            messages = ["message 1", "message 2", "message 3"]
            answer = first(messages)
            assert answer == messages[0]
    """
    return messages[0]


def nop(messages: list[Any]) -> Any:
    """No change after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from byte.processor.post import nop

            messages = ["message 1", "message 2", "message 3"]
            answer = nop(messages)
            assert answer == messages
    """
    return messages


def temperature_softmax(messages: list[Any], scores: list[float], temperature: float = 0.0) -> Any:
    """Post processing with temperature softmax after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]
    :param scores: A list of evaluation scores corresponding to `messages`
    :type scores: List[float]
    :param temperature: A non-negative number of sampling temperature, defaults to 0.
                        A higher temperature makes the output more random.
                        A lower temperature means a more deterministic and confident output.
    :type temperature: float

    Example:
        .. code-block:: python

            from byte.processor.post import temperature_softmax

            messages = ["message 1", "message 2", "message 3"]
            scores = [0.9, 0.5, 0.1]
            answer = temperature_softmax(messages, scores, temperature=0.5)
    """

    if temperature > 0:
        scores = softmax([x / temperature for x in scores])
        return numpy.random.choice(messages, size=1, p=scores)[0]

    m_s = list(zip(messages, scores))
    return sorted(m_s, key=lambda x: x[1], reverse=True)[0][0]


def llm_semantic_verification(
    messages: list[Any],
    scores: list[float] | None = None,
    original_question: str | None = None,
    *,
    client=None,
    system_prompt: str | None = None,
    model: str = "gpt-3.5-turbo",
    **kwargs,
) -> Any:
    """
    Use LLM to verify whether the answer is semantically consistent with the question.
    If the answer passes verification, return it; otherwise, return None (to trigger a real LLM call).

    :param messages: A list of candidate outputs.
    :type messages: List[Any]
    :param scores: A list of evaluation scores corresponding to messages.
    :type scores: List[float], optional
    :param original_question: The original question string.
    :type original_question: str, optional
    :param client: LLM client object, defaults to None.
    :type client: Any, optional
    :param system_prompt: System prompt, defaults to None.
    :type system_prompt: str, optional
    :param model: LLM model name, defaults to "gpt-3.5-turbo".
    :type model: str, optional
    :param temperature: Sampling temperature, defaults to 0.0.
    :type temperature: float, optional
    :param kwargs: Other keyword arguments.
    :return: The answer if it passes semantic verification, otherwise None.
    :rtype: Any

    Example:
        .. code-block:: python

            from byte.processor.post import llm_semantic_verification

            messages = ["answer1", "answer2"]
            scores = [0.9, 0.5]
            question = "original question"
            answer = llm_semantic_verification(messages, scores, original_question=question)
    """
    if not messages or not original_question:
        return None

    best_answer = messages[0] if not scores else messages[scores.index(max(scores))]
    if system_prompt is None:
        system_prompt = (
            "You are a strict semantic verification assistant. "
            "Only answer 'yes' or 'no'. If unsure, answer 'no'."
        )

    try:
        completion_create = _resolve_completion_create(client)
        resp = completion_create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question: {original_question}\n"
                        f"Answer: {best_answer}\n"
                        "Does this answer fully match the question? yes/no"
                    ),
                },
            ],
            temperature=0,
            max_tokens=10,
            **kwargs,
        )
        verdict = _extract_completion_text(resp).strip().lower()
        if verdict == "yes":
            return best_answer
    except Exception as e:  # pragma: no cover - defensive logging
        byte_log.warning("LLM verification failed: %s", e)

    return None


class LlmVerifier:
    """
    LlmVerifier is a callable class that wraps the llm_semantic_verification function.
    It stores the LLM client, system prompt, and model name for repeated semantic verification tasks.

    :param client: LLM client object.
    :type client: Any
    :param system_prompt: System prompt for the LLM.
    :type system_prompt: str
    :param model: LLM model name, defaults to "gpt-3.5-turbo".
    :type model: str, optional
    """

    def __init__(self, client=None, system_prompt=None, model="gpt-3.5-turbo") -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.model = model

    def __call__(self, messages, scores=None, original_question=None, **kwargs) -> Any:
        """
        Call the verifier to perform semantic verification using the stored client, prompt, and model.

        :param messages: A list of candidate outputs.
        :param scores: A list of evaluation scores corresponding to messages.
        :param original_question: The original question string.
        :param temperature: Sampling temperature.
        :param kwargs: Other keyword arguments.
        :return: The answer if it passes semantic verification, otherwise None.
        """
        return llm_semantic_verification(
            messages,
            scores=scores,
            original_question=original_question,
            client=self.client,
            system_prompt=self.system_prompt,
            model=self.model,
            **kwargs,
        )
