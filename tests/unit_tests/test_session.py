"""Session-scoped cache behavior tests with optional OpenAI dependency handling."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from byte._testing.optional_deps import skip_module_if_feature_missing

skip_module_if_feature_missing("onnx", "openai")

from byte import cache
from byte._backends import openai
from byte.embedding import Onnx
from byte.manager import manager_factory
from byte.processor.pre import get_prompt
from byte.session import Session
from byte.similarity_evaluation.distance import SearchDistanceEvaluation
from byte.utils.response import get_text_from_openai_answer


def check_hit(cur_session_id, cache_session_ids, cache_questions, cache_answer) -> object:
    return bool(cache_questions and "what" in cache_questions[0])


def _real_openai() -> object:
    try:
        import openai as real_openai  # pylint: disable=C0415
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("openai is not installed") from exc
    return real_openai


class TestSession(unittest.TestCase):
    """Test Session"""

    question = "what is your name?"
    expect_answer = "byte"
    session_id = "test_map"

    def _completion_payload(self) -> object:
        return {
            "choices": [{"text": self.expect_answer, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "text-davinci-003",
            "object": "text_completion",
        }

    def test_with(self) -> None:
        with TemporaryDirectory() as session_dir:
            data_manager = manager_factory("map", data_dir=session_dir)
            cache.init(data_manager=data_manager, pre_embedding_func=get_prompt)

            session0 = Session(self.session_id, check_hit_func=check_hit)
            self.assertEqual(session0.name, self.session_id)
            with patch.object(
                openai.Completion, "llm", return_value=self._completion_payload()
            ), Session() as session:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=self.question,
                    session=session,
                )
                answer_text = get_text_from_openai_answer(response)
                self.assertEqual(answer_text, self.expect_answer)
            self.assertEqual(len(data_manager.list_sessions()), 0)

    def test_map(self) -> None:
        with TemporaryDirectory() as session_dir:
            data_manager = manager_factory("map", data_dir=session_dir)
            cache.init(data_manager=data_manager, pre_embedding_func=get_prompt)

            session0 = Session(self.session_id, check_hit_func=check_hit)
            with patch.object(openai.Completion, "llm", return_value=self._completion_payload()):
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=self.question,
                    session=session0,
                )
                answer_text = get_text_from_openai_answer(response)
                self.assertEqual(answer_text, self.expect_answer)

            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=self.question,
                session=session0,
            )
            answer_text = get_text_from_openai_answer(response)
            self.assertEqual(answer_text, self.expect_answer)

            session1 = Session()
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=self.question,
                session=session1,
            )
            answer_text = get_text_from_openai_answer(response)
            self.assertEqual(answer_text, self.expect_answer)

            with self.assertRaises(_real_openai().OpenAIError):
                openai.Completion.create(
                    model="text-davinci-003",
                    prompt=self.question,
                    session=session1,
                )

            self.assertEqual(len(data_manager.list_sessions()), 2)
            session0.drop()
            session1.drop()
            self.assertEqual(len(data_manager.list_sessions()), 0)

    def test_ssd(self) -> None:
        with TemporaryDirectory() as session_dir:
            onnx = Onnx()
            try:
                onnx_dimension = onnx.dimension
            except (ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise unittest.SkipTest(
                    f"onnx optional feature stack is not usable in this environment: {exc}"
                ) from exc
            data_manager = manager_factory(
                "sqlite,faiss",
                session_dir,
                vector_params={"dimension": onnx_dimension},
            )
            cache.init(
                pre_embedding_func=get_prompt,
                embedding_func=onnx.to_embeddings,
                data_manager=data_manager,
                similarity_evaluation=SearchDistanceEvaluation(),
            )

            session0 = Session(self.session_id, check_hit_func=check_hit)
            with patch.object(openai.Completion, "llm", return_value=self._completion_payload()):
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=self.question,
                    session=session0,
                )
                answer_text = get_text_from_openai_answer(response)
                self.assertEqual(answer_text, self.expect_answer)

            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=self.question,
                session=session0,
            )
            answer_text = get_text_from_openai_answer(response)
            self.assertEqual(answer_text, self.expect_answer)

            session1 = Session()
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=self.question,
                session=session1,
            )
            answer_text = get_text_from_openai_answer(response)
            self.assertEqual(answer_text, self.expect_answer)

            with self.assertRaises(_real_openai().OpenAIError):
                openai.Completion.create(
                    model="text-davinci-003",
                    prompt=self.question,
                    session=session1,
                )

            self.assertEqual(len(data_manager.list_sessions()), 2)
            session0.drop()
            session1.drop()
            self.assertEqual(len(data_manager.list_sessions()), 0)
