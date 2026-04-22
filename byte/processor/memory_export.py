import csv
import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

from byte.security import SensitiveDataProtector, read_validated_zip_member

_ENTRY_SECTIONS = (
    "tool_results",
    "ai_memory",
    "execution_memory",
    "failure_memory",
    "patch_patterns",
    "prompt_pieces",
    "artifact_memory",
    "workflow_plans",
    "session_deltas",
)


def export_snapshot_artifact(
    snapshot: dict[str, Any],
    path: str,
    *,
    format: str | None = None,
    encryption_key: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    resolved_format = _resolve_format(target, format)
    target.parent.mkdir(parents=True, exist_ok=True)
    protector = SensitiveDataProtector(encryption_key)

    if protector.enabled:
        raw_bytes = _render_artifact_bytes(snapshot, resolved_format)
        envelope = {
            "byte_secure_artifact": True,
            "artifact_format": resolved_format,
            "algorithm": "fernet",
            "ciphertext": protector.encrypt_bytes(raw_bytes).decode("ascii"),
        }
        target.write_text(json.dumps(envelope, ensure_ascii=True, indent=2), encoding="utf-8")
    else:
        _write_artifact(target, snapshot, resolved_format)

    return {
        "path": str(target),
        "format": f"encrypted:{resolved_format}" if protector.enabled else resolved_format,
        "rows": len(_snapshot_rows(snapshot)),
        "entry_sections": list(_ENTRY_SECTIONS),
    }


def load_snapshot_artifact(
    path: str,
    *,
    format: str | None = None,
    encryption_key: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    resolved_format = _resolve_format(target, format)
    raw_bytes = target.read_bytes()
    maybe_envelope = _maybe_parse_encrypted_envelope(raw_bytes)
    if maybe_envelope is not None:
        protector = SensitiveDataProtector(encryption_key)
        if not protector.enabled:
            raise ValueError(
                "This memory artifact is encrypted. Provide an encryption key to import it."
            )
        decrypted = protector.decrypt_bytes(maybe_envelope["ciphertext"].encode("ascii"))
        return _load_from_bytes(decrypted, maybe_envelope["artifact_format"])
    return _load_from_bytes(raw_bytes, resolved_format)


def _resolve_format(target: Path, override: str | None) -> str:
    if override:
        return str(override).strip().lower()
    suffix = target.suffix.lower().lstrip(".")
    return suffix or "json"


def _write_artifact(target: Path, snapshot: dict[str, Any], resolved_format: str) -> None:
    if resolved_format == "json":
        target.write_text(
            json.dumps(snapshot, ensure_ascii=True, indent=2, default=str), encoding="utf-8"
        )
    elif resolved_format == "csv":
        _write_csv(target, snapshot)
    elif resolved_format == "zip":
        _write_zip(target, snapshot)
    elif resolved_format == "parquet":
        _write_parquet(target, snapshot)
    elif resolved_format in {"sqlite", "db"}:
        _write_sqlite(target, snapshot)
    else:
        raise ValueError(f"Unsupported memory export format: {resolved_format}")


def _render_artifact_bytes(snapshot: dict[str, Any], resolved_format: str) -> bytes:
    if resolved_format == "json":
        return json.dumps(snapshot, ensure_ascii=True, indent=2, default=str).encode("utf-8")
    if resolved_format == "csv":
        rows = _snapshot_rows(snapshot)
        if not rows:
            return b""
        buffer = []
        fieldnames = list(rows[0].keys())
        buffer.append(",".join(fieldnames))
        for row in rows:
            buffer.append(
                ",".join(_csv_escape(str(row.get(field, "") or "")) for field in fieldnames)
            )
        return "\n".join(buffer).encode("utf-8")
    if resolved_format == "zip":
        import io  # pylint: disable=C0415

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "snapshot.json", json.dumps(snapshot, ensure_ascii=True, indent=2, default=str)
            )
            rows = _snapshot_rows(snapshot)
            if rows:
                fieldnames = list(rows[0].keys())
                csv_lines = [",".join(fieldnames)]
                for row in rows:
                    csv_lines.append(
                        ",".join(_csv_escape(str(row.get(field, "") or "")) for field in fieldnames)
                    )
                archive.writestr("snapshot.csv", "\n".join(csv_lines))
        return buffer.getvalue()
    if resolved_format == "parquet":
        import tempfile  # pylint: disable=C0415

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "snapshot.parquet"
            _write_parquet(temp_path, snapshot)
            return temp_path.read_bytes()
    if resolved_format in {"sqlite", "db"}:
        import tempfile  # pylint: disable=C0415

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "snapshot.sqlite"
            _write_sqlite(temp_path, snapshot)
            return temp_path.read_bytes()
    raise ValueError(f"Unsupported memory export format: {resolved_format}")


def _load_from_bytes(raw_bytes: bytes, resolved_format: str) -> dict[str, Any]:
    if resolved_format == "json":
        return json.loads(raw_bytes.decode("utf-8"))
    if resolved_format == "zip":
        return json.loads(read_validated_zip_member(raw_bytes, "snapshot.json").decode("utf-8"))
    if resolved_format == "csv":
        import io  # pylint: disable=C0415

        return _snapshot_from_rows(list(csv.DictReader(io.StringIO(raw_bytes.decode("utf-8")))))
    if resolved_format == "parquet":
        import tempfile  # pylint: disable=C0415

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "snapshot.parquet"
            temp_path.write_bytes(raw_bytes)
            return _snapshot_from_rows(_read_parquet(temp_path))
    if resolved_format in {"sqlite", "db"}:
        import tempfile  # pylint: disable=C0415

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "snapshot.sqlite"
            temp_path.write_bytes(raw_bytes)
            return _read_sqlite(temp_path)
    raise ValueError(f"Unsupported memory import format: {resolved_format}")


def _maybe_parse_encrypted_envelope(raw_bytes: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("byte_secure_artifact"):
        return None
    return {
        "artifact_format": str(payload.get("artifact_format", "") or ""),
        "ciphertext": str(payload.get("ciphertext", "") or ""),
    }


def _snapshot_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "row_type": "meta",
            "section": "memory_scope",
            "key": "",
            "provider": "",
            "model": "",
            "category": "",
            "route_key": "",
            "canonical_key": "",
            "question": "",
            "answer": "",
            "verified": "",
            "hits": "",
            "created_at": "",
            "updated_at": "",
            "payload_json": json.dumps(
                {"memory_scope": snapshot.get("memory_scope", "")}, sort_keys=True, default=str
            ),
        },
        {
            "row_type": "meta",
            "section": "intent_graph",
            "key": "",
            "provider": "",
            "model": "",
            "category": "",
            "route_key": "",
            "canonical_key": "",
            "question": "",
            "answer": "",
            "verified": "",
            "hits": "",
            "created_at": "",
            "updated_at": "",
            "payload_json": json.dumps(
                snapshot.get("intent_graph", {}) or {}, sort_keys=True, default=str
            ),
        },
    ]
    for section in _ENTRY_SECTIONS:
        payload = snapshot.get(section, {}) or {}
        rows.append(
            {
                "row_type": "meta",
                "section": f"{section}:stats",
                "key": "",
                "provider": "",
                "model": "",
                "category": "",
                "route_key": "",
                "canonical_key": "",
                "question": "",
                "answer": "",
                "verified": "",
                "hits": "",
                "created_at": "",
                "updated_at": "",
                "payload_json": json.dumps(
                    payload.get("stats", {}) or {}, sort_keys=True, default=str
                ),
            }
        )
        for entry in payload.get("entries", []) or []:
            rows.append(
                {
                    "row_type": "entry",
                    "section": section,
                    "key": str(entry.get("key", "") or ""),
                    "provider": str(entry.get("provider", "") or ""),
                    "model": str(entry.get("model", "") or ""),
                    "category": str(entry.get("category", "") or ""),
                    "route_key": str(entry.get("route_key", "") or ""),
                    "canonical_key": str(entry.get("canonical_key", "") or ""),
                    "question": str(entry.get("question", "") or ""),
                    "answer": _stringify_value(entry.get("answer")),
                    "verified": _stringify_value(entry.get("verified")),
                    "hits": _stringify_value(entry.get("hits")),
                    "created_at": _stringify_value(entry.get("created_at")),
                    "updated_at": _stringify_value(entry.get("updated_at")),
                    "payload_json": json.dumps(entry, sort_keys=True, default=str),
                }
            )
    return rows


def _snapshot_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot = {
        "memory_scope": "",
        "intent_graph": {},
    }
    for section in _ENTRY_SECTIONS:
        snapshot[section] = {"entries": [], "stats": {}}

    for row in rows:
        section = str(row.get("section", "") or "")
        payload = json.loads(str(row.get("payload_json", "") or "{}"))
        if row.get("row_type") == "meta":
            if section == "memory_scope":
                snapshot["memory_scope"] = str(payload.get("memory_scope", "") or "")
            elif section == "intent_graph":
                snapshot["intent_graph"] = payload
            elif section.endswith(":stats"):
                snapshot[section.split(":", 1)[0]]["stats"] = payload
            continue
        if section in snapshot:
            snapshot[section]["entries"].append(payload)
    return snapshot


def _write_csv(target: Path, snapshot: dict[str, Any]) -> None:
    rows = _snapshot_rows(snapshot)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(target: Path) -> list[dict[str, Any]]:
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_zip(target: Path, snapshot: dict[str, Any]) -> None:
    rows = _snapshot_rows(snapshot)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "snapshot.json", json.dumps(snapshot, ensure_ascii=True, indent=2, default=str)
        )
        if rows:
            buffer = []
            fieldnames = list(rows[0].keys())
            buffer.append(",".join(fieldnames))
            for row in rows:
                buffer.append(
                    ",".join(_csv_escape(str(row.get(field, "") or "")) for field in fieldnames)
                )
            archive.writestr("snapshot.csv", "\n".join(buffer))


def _write_parquet(target: Path, snapshot: dict[str, Any]) -> None:
    rows = _snapshot_rows(snapshot)
    try:
        import pandas as pd  # pylint: disable=C0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Parquet export requires pandas and a parquet engine such as pyarrow or fastparquet."
        ) from exc
    frame = pd.DataFrame(rows)
    try:
        frame.to_parquet(target, index=False)
    except Exception as exc:  # pylint: disable=W0703
        raise ImportError(
            "Parquet export requires a parquet engine. Install pyarrow or fastparquet to enable it."
        ) from exc


def _read_parquet(target: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # pylint: disable=C0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Parquet import requires pandas and a parquet engine.") from exc
    frame = pd.read_parquet(target)
    return frame.fillna("").to_dict(orient="records")


def _write_sqlite(target: Path, snapshot: dict[str, Any]) -> None:
    rows = _snapshot_rows(snapshot)
    connection = sqlite3.connect(str(target))
    try:
        cursor = connection.cursor()
        cursor.execute("drop table if exists snapshot_rows")
        cursor.execute(
            """
            create table snapshot_rows (
                row_type text not null,
                section text not null,
                key text,
                provider text,
                model text,
                category text,
                route_key text,
                canonical_key text,
                question text,
                answer text,
                verified text,
                hits text,
                created_at text,
                updated_at text,
                payload_json text not null
            )
            """
        )
        cursor.executemany(
            """
            insert into snapshot_rows (
                row_type, section, key, provider, model, category, route_key, canonical_key,
                question, answer, verified, hits, created_at, updated_at, payload_json
            ) values (
                :row_type, :section, :key, :provider, :model, :category, :route_key, :canonical_key,
                :question, :answer, :verified, :hits, :created_at, :updated_at, :payload_json
            )
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def _read_sqlite(target: Path) -> dict[str, Any]:
    connection = sqlite3.connect(str(target))
    try:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in connection.execute("select * from snapshot_rows order by rowid asc")
        ]
        return _snapshot_from_rows(rows)
    finally:
        connection.close()


def _stringify_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list, tuple, bool)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _csv_escape(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'
