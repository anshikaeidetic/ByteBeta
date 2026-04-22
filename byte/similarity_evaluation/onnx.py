from typing import Any

import numpy as np

from byte.similarity_evaluation import SimilarityEvaluation
from byte.utils import lazy_optional_module
from byte.utils.log import byte_log

onnxruntime = lazy_optional_module("onnxruntime", package="onnxruntime")
huggingface_hub = lazy_optional_module("huggingface_hub", package="huggingface-hub")
transformers = lazy_optional_module("transformers", package="transformers")


def pad_sequence(input_ids_list: list[np.ndarray], padding_value: int = 0) -> Any:
    max_len = max(len(sequence) for sequence in input_ids_list)
    padded_sequences = np.full((len(input_ids_list), max_len), padding_value, dtype=np.int64)
    for i, sequence in enumerate(input_ids_list):
        padded_sequences[i, : len(sequence)] = sequence
    return padded_sequences


class OnnxModelEvaluation(SimilarityEvaluation):
    """Using ONNX model to evaluate sentences pair similarity.

    This evaluator use the ONNX model to evaluate the similarity of two sentences.

    :param model: model name of OnnxModelEvaluation. Default is 'GPTCache/albert-duplicate-onnx'.
    :type model: str

    Example:
        .. code-block:: python

            from byte.similarity_evaluation import OnnxModelEvaluation

            evaluation = OnnxModelEvaluation()
            score = evaluation.evaluation(
                {
                    'question': 'What is the color of sky?'
                },
                {
                    'question': 'hello'
                }
            )
    """

    def __init__(self, model: str = "GPTCache/albert-duplicate-onnx") -> None:
        self.model = model
        self._tokenizer_name = "albert-base-v2"
        self.tokenizer = None
        self.ort_session = None

    def _ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.ort_session is not None:
            return
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self._tokenizer_name)
        onnx_model_path = huggingface_hub.hf_hub_download(repo_id=self.model, filename="model.onnx")
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # WARNING: the model cannot evaluate text with more than 512 tokens
    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **_) -> float:
        """Evaluate the similarity score of pair.

        :param src_dict: the query dictionary to evaluate with cache.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]
            if src_question.lower() == cache_question.lower():
                return 1
            return self.inference(src_question, [cache_question])
        except Exception as e:  # pylint: disable=W0703
            byte_log.warning(
                "OnnxModelEvaluation failed for [%s] vs [%s]: %s",
                src_dict.get("question", "?"),
                cache_dict.get("question", "?"),
                e,
            )
            return 0

    def range(self) -> tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 1.0

    def inference(self, reference: str, candidates: list[str]) -> np.ndarray:
        """Inference the ONNX model.

        :param reference: reference sentence.
        :type reference: str
        :param candidates: candidate sentences.
        :type candidates: List[str]

        :return: probability score indcates how much is reference similar to candidates.
        """
        self._ensure_loaded()
        n_candidates = len(candidates)
        inference_texts = [{"text_a": reference, "text_b": candidate} for candidate in candidates]
        batch_encoding_list = [
            self.tokenizer(
                text["text_a"],
                text["text_b"],
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            for text in inference_texts
        ]

        input_ids_list = [
            np.asarray(encode["input_ids"], dtype=np.int64) for encode in batch_encoding_list
        ]
        attention_mask_list = [
            np.asarray(encode["attention_mask"], dtype=np.int64) for encode in batch_encoding_list
        ]
        token_type_ids_list = [
            np.asarray(
                encode.get("token_type_ids", np.zeros(len(encode["input_ids"]), dtype=np.int64)),
                dtype=np.int64,
            )
            for encode in batch_encoding_list
        ]

        padded_input_ids = pad_sequence(input_ids_list, padding_value=self.tokenizer.pad_token_id)
        padded_attention_mask = pad_sequence(
            attention_mask_list, padding_value=self.tokenizer.pad_token_id
        )
        padded_token_type_ids = pad_sequence(
            token_type_ids_list, padding_value=self.tokenizer.pad_token_id
        )

        ort_inputs = {
            "input_ids": padded_input_ids.reshape(n_candidates, -1),
            "attention_mask": padded_attention_mask.reshape(n_candidates, -1),
            "token_type_ids": padded_token_type_ids.reshape(n_candidates, -1),
        }
        ort_outputs = self.ort_session.run(None, ort_inputs)
        scores = ort_outputs[0][:, 1]
        return float(scores[0])
