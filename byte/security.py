import base64
import copy
import hashlib
import io
import json
import os
import zipfile
from contextlib import contextmanager
from ipaddress import ip_address
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken

from byte.manager.scalar_data.base import Answer, DataType, Question, QuestionDep
from byte.utils.error import CacheError

_ENC_PREFIX = "byte-sec:v1:"
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
DEFAULT_SECURITY_MAX_REQUEST_BYTES = 1_048_576
DEFAULT_SECURITY_MAX_UPLOAD_BYTES = 16_777_216
DEFAULT_SECURITY_MAX_ARCHIVE_BYTES = 33_554_432
DEFAULT_SECURITY_MAX_ARCHIVE_MEMBERS = 256


def _normalize_key_material(value: str | None) -> bytes | None:
    raw = str(value or os.getenv("BYTE_SECURITY_KEY") or "").strip()
    if not raw:
        return None
    candidate = raw.encode("utf-8")
    if len(candidate) == 44:
        return candidate
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def short_digest(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def redact_text(value: Any) -> str:
    raw = "" if value is None else str(value)
    return f"[redacted len={len(raw)} sha256={short_digest(raw)}]"


def sanitize_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_structure(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_structure(item) for item in value]
    if isinstance(value, set):
        return sorted(sanitize_structure(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def sanitize_request_preview(request_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    request_kwargs = dict(request_kwargs or {})
    preview = {
        "model": str(request_kwargs.get("model", "") or ""),
        "has_messages": bool(request_kwargs.get("messages")),
        "message_count": len(request_kwargs.get("messages", []) or []),
        "has_tools": bool(request_kwargs.get("tools")),
        "prompt_digest": short_digest(
            json.dumps(
                request_kwargs.get(
                    "messages", request_kwargs.get("prompt", request_kwargs.get("input", ""))
                ),
                sort_keys=True,
                default=str,
            )
        ),
    }
    return preview


def _normalize_allowed_hosts(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    return [str(item).strip().lower() for item in value if str(item).strip()]


def _is_loopback_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if normalized in _LOOPBACK_HOSTS:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _host_matches_allowed(host: str, allowed_hosts: list[str]) -> bool:
    normalized = str(host or "").strip().lower()
    for allowed in allowed_hosts:
        if not allowed:
            continue
        if allowed.startswith("*."):
            suffix = allowed[1:]
            if normalized.endswith(suffix) and normalized != suffix.lstrip("."):
                return True
            continue
        if normalized == allowed:
            return True
    return False


def validate_outbound_target(target: str, *, allowed_hosts: Any | None = None) -> str:
    candidate = str(target or "").strip()
    if not candidate:
        raise CacheError("Provider host override cannot be empty.")
    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise CacheError("Provider host overrides must use http or https URLs.")
    if not parsed.hostname:
        raise CacheError("Provider host overrides must include a valid hostname.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CacheError(
            "Provider host overrides cannot include credentials, query strings, or fragments."
        )

    host = str(parsed.hostname or "").strip().lower()
    allowlist = _normalize_allowed_hosts(allowed_hosts)
    if allowlist:
        if not _host_matches_allowed(host, allowlist):
            raise CacheError(
                "Provider host override target is not in Byte's allowed egress host list."
            )
    elif not _is_loopback_host(host):
        raise CacheError(
            "Provider host overrides are limited to loopback targets unless an allowlist is configured."
        )

    if parsed.scheme.lower() != "https" and not _is_loopback_host(host):
        raise CacheError("Non-loopback provider host overrides must use HTTPS.")

    return candidate


def ensure_byte_limit(size: int, *, limit: int, label: str) -> None:
    bounded_limit = max(1, int(limit))
    if int(size) > bounded_limit:
        raise CacheError(f"{label} exceeds Byte's configured size limit of {bounded_limit} bytes.")


def validate_declared_content_length(
    content_length: int | None, *, limit: int, label: str
) -> None:
    if content_length in (None, ""):
        return
    ensure_byte_limit(int(content_length), limit=limit, label=label)


def _normalize_archive_member_name(name: str) -> str:
    normalized = PurePosixPath(str(name or ""))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise CacheError("Archive contains an invalid member path.")
    member_name = str(normalized).strip()
    if not member_name:
        raise CacheError("Archive contains an empty member path.")
    return member_name


@contextmanager
def validated_zip_archive(
    raw_bytes: bytes,
    *,
    max_archive_bytes: int = DEFAULT_SECURITY_MAX_ARCHIVE_BYTES,
    max_member_count: int = DEFAULT_SECURITY_MAX_ARCHIVE_MEMBERS,
    required_members: list[str] | None = None,
) -> Any:
    ensure_byte_limit(len(raw_bytes), limit=max_archive_bytes, label="Archive payload")
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as archive:
            members = archive.infolist()
            if len(members) > int(max_member_count):
                raise CacheError(
                    "Archive exceeds Byte's configured member count limit."
                )
            total_uncompressed = 0
            normalized_names: set[str] = set()
            for info in members:
                member_name = _normalize_archive_member_name(info.filename)
                normalized_names.add(member_name)
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    raise CacheError("Encrypted ZIP artifacts are not supported.")
                total_uncompressed += max(0, int(info.file_size))
                ensure_byte_limit(
                    total_uncompressed,
                    limit=max_archive_bytes,
                    label="Archive contents",
                )
            for required in required_members or []:
                if _normalize_archive_member_name(required) not in normalized_names:
                    raise CacheError(f"Archive is missing required member '{required}'.")
            yield archive
    except zipfile.BadZipFile as exc:
        raise CacheError("Artifact archive is not a valid ZIP file.") from exc


def read_validated_zip_member(
    raw_bytes: bytes,
    member_name: str,
    *,
    max_archive_bytes: int = DEFAULT_SECURITY_MAX_ARCHIVE_BYTES,
    max_member_count: int = DEFAULT_SECURITY_MAX_ARCHIVE_MEMBERS,
) -> bytes:
    normalized_member = _normalize_archive_member_name(member_name)
    with validated_zip_archive(
        raw_bytes,
        max_archive_bytes=max_archive_bytes,
        max_member_count=max_member_count,
        required_members=[normalized_member],
    ) as archive:
        payload = archive.read(normalized_member)
    ensure_byte_limit(len(payload), limit=max_archive_bytes, label=f"Archive member {normalized_member}")
    return payload


def sanitize_outbound_overrides(request_kwargs: dict[str, Any], config: Any) -> dict[str, Any]:
    payload = dict(request_kwargs or {})
    override_fields = [
        field for field in ("api_base", "base_url", "host") if payload.get(field) not in (None, "")
    ]
    if not override_fields:
        return payload
    if not getattr(config, "security_mode", False):
        return payload

    allow_override = bool(getattr(config, "security_allow_provider_host_override", False))
    if not allow_override:
        joined = ", ".join(sorted(override_fields))
        raise CacheError(
            f"Client-supplied provider host overrides are disabled by Byte security policy: {joined}."
        )

    allowed_hosts = getattr(config, "security_allowed_egress_hosts", None)
    for field in override_fields:
        payload[field] = validate_outbound_target(payload[field], allowed_hosts=allowed_hosts)

    return payload


class SensitiveDataProtector:
    def __init__(self, key: str | None = None) -> None:
        normalized = _normalize_key_material(key)
        self._fernet = Fernet(normalized) if normalized is not None else None

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def encrypt_text(self, value: Any) -> Any:
        if not self.enabled or value in (None, ""):
            return value
        text = str(value)
        if text.startswith(_ENC_PREFIX):
            return text
        token = self._fernet.encrypt(text.encode("utf-8")).decode("ascii")
        return _ENC_PREFIX + token

    def decrypt_text(self, value: Any) -> Any:
        if not self.enabled or not isinstance(value, str) or not value.startswith(_ENC_PREFIX):
            return value
        token = value[len(_ENC_PREFIX) :].encode("ascii")
        try:
            return self._fernet.decrypt(token).decode("utf-8")
        except InvalidToken:
            return value

    def encrypt_bytes(self, value: bytes) -> bytes:
        if not self.enabled or value in (None, b""):
            return value
        return self._fernet.encrypt(value)

    def decrypt_bytes(self, value: bytes) -> bytes:
        if not self.enabled or value in (None, b""):
            return value
        try:
            return self._fernet.decrypt(value)
        except InvalidToken:
            return value


class SecureDataManager:
    """Wrap a data manager with encryption-at-rest and redacted reporting."""

    def __init__(
        self, delegate, *, encryption_key: str | None = None, redact_reports: bool = True
    ) -> None:
        self.delegate = delegate
        self.protector = SensitiveDataProtector(encryption_key)
        self.redact_reports = bool(redact_reports)
        if hasattr(delegate, "data"):
            self.data = delegate.data
        if hasattr(delegate, "s"):
            self.s = delegate.s
        if hasattr(delegate, "v"):
            self.v = delegate.v
        if hasattr(delegate, "o"):
            self.o = delegate.o

    def __getattr__(self, item) -> Any:
        return getattr(self.delegate, item)

    def save(self, question, answer, embedding_data, **kwargs) -> Any:
        return self.delegate.save(
            self._encode_question(question),
            self._encode_answers(answer),
            embedding_data,
            **kwargs,
        )

    def import_data(self, questions, answers, embedding_datas, session_ids) -> Any:
        safe_questions = [self._encode_question(question) for question in questions]
        safe_answers = [self._encode_answers(answer) for answer in answers]
        return self.delegate.import_data(safe_questions, safe_answers, embedding_datas, session_ids)

    def get_scalar_data(self, res_data, **kwargs) -> Any | None:
        cache_data = self.delegate.get_scalar_data(res_data, **kwargs)
        if cache_data is None:
            return None
        return self._decode_cache_data(cache_data)

    def search(self, embedding_data, **kwargs) -> Any:
        return self.delegate.search(embedding_data, **kwargs)

    def flush(self) -> Any:
        return self.delegate.flush()

    def add_session(self, res_data, session_id, pre_embedding_data) -> Any:
        return self.delegate.add_session(res_data, session_id, pre_embedding_data)

    def list_sessions(self, session_id=None, key=None) -> Any:
        return self.delegate.list_sessions(session_id, key)

    def delete_session(self, session_id) -> Any:
        return self.delegate.delete_session(session_id)

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> Any:
        if not self.redact_reports:
            return self.delegate.report_cache(
                user_question,
                cache_question,
                cache_question_id,
                cache_answer,
                similarity_value,
                cache_delta_time,
            )
        return self.delegate.report_cache(
            redact_text(user_question),
            redact_text(cache_question),
            cache_question_id,
            redact_text(cache_answer),
            similarity_value,
            cache_delta_time,
        )

    def close(self) -> Any:
        return self.delegate.close()

    def invalidate_by_query(self, query: str, *, embedding_func=None) -> bool:
        if hasattr(self.delegate, "data") and hasattr(self.delegate.data, "items"):
            for key, value in list(self.delegate.data.items()):
                stored_question = value[0]
                decoded = self._decode_question(stored_question)
                compare_value = decoded.content if isinstance(decoded, Question) else decoded
                if compare_value == query:
                    del self.delegate.data[key]
                    return True
            return False
        if embedding_func is None:
            return False
        embedding = embedding_func(query)
        results = self.delegate.search(embedding)
        if results:
            best_id = results[0][1]
            if hasattr(self.delegate, "s"):
                self.delegate.s.delete([best_id])
            if hasattr(self.delegate, "v"):
                self.delegate.v.delete([best_id])
            return True
        return False

    def _encode_question(self, question) -> Any:
        if isinstance(question, Question):
            deps = []
            for dep in question.deps or []:
                dep_data = dep.data
                if dep.dep_type == DataType.STR and isinstance(dep_data, str):
                    dep_data = self.protector.encrypt_text(dep_data)
                deps.append(QuestionDep(name=dep.name, data=dep_data, dep_type=dep.dep_type))
            return Question(
                content=self.protector.encrypt_text(question.content),
                deps=deps if question.deps is not None else None,
            )
        if isinstance(question, str):
            return self.protector.encrypt_text(question)
        return question

    def _encode_answers(self, answers) -> Any:
        if isinstance(answers, list):
            return [self._encode_answers(answer) for answer in answers]
        if isinstance(answers, tuple):
            return tuple(self._encode_answers(answer) for answer in answers)
        if isinstance(answers, Answer):
            answer_value = answers.answer
            if answers.answer_type == DataType.STR and isinstance(answer_value, str):
                answer_value = self.protector.encrypt_text(answer_value)
            return Answer(answer_value, answers.answer_type)
        if isinstance(answers, str):
            return self.protector.encrypt_text(answers)
        return answers

    def _decode_question(self, question) -> Any:
        if isinstance(question, Question):
            deps = []
            for dep in question.deps or []:
                dep_data = dep.data
                if dep.dep_type == DataType.STR and isinstance(dep_data, str):
                    dep_data = self.protector.decrypt_text(dep_data)
                deps.append(QuestionDep(name=dep.name, data=dep_data, dep_type=dep.dep_type))
            return Question(
                content=self.protector.decrypt_text(question.content),
                deps=deps if question.deps is not None else None,
            )
        if isinstance(question, str):
            return self.protector.decrypt_text(question)
        return question

    def _decode_cache_data(self, cache_data) -> Any:
        cache_data.question = self._decode_question(cache_data.question)
        for answer in cache_data.answers:
            if answer.answer_type == DataType.STR and isinstance(answer.answer, str):
                answer.answer = self.protector.decrypt_text(answer.answer)
        return cache_data


def maybe_wrap_data_manager(data_manager, config) -> Any:
    if isinstance(data_manager, SecureDataManager):
        return data_manager
    if (
        not getattr(config, "security_mode", False)
        and not getattr(config, "security_encryption_key", None)
        and not getattr(config, "security_redact_reports", False)
    ):
        return data_manager
    return SecureDataManager(
        data_manager,
        encryption_key=getattr(config, "security_encryption_key", None),
        redact_reports=getattr(config, "security_redact_reports", True),
    )


def redact_memory_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(snapshot or {})
    for section in (
        "tool_results",
        "ai_memory",
        "execution_memory",
        "failure_memory",
        "patch_patterns",
        "prompt_pieces",
        "artifact_memory",
        "workflow_plans",
        "session_deltas",
    ):
        payload = redacted.get(section, {}) or {}
        entries = payload.get("entries", []) or []
        for entry in entries:
            for key in (
                "question",
                "answer",
                "reasoning",
                "tool_outputs",
                "patch",
                "test_command",
                "test_result",
                "lint_result",
                "schema_validation",
                "tool_checks",
                "metadata",
                "preview",
                "summary",
            ):
                if key in entry:
                    entry[key] = sanitize_structure(entry.get(key))
        redacted[section] = payload
    return redacted
