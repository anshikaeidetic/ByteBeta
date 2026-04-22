import json
import uuid
import zipfile

import pytest

from byte import Cache, Config
from byte.adapter.api import export_memory_artifact, import_memory_artifact, remember_interaction
from byte.manager.factory import get_data_manager
from byte.processor.memory_export import load_snapshot_artifact
from byte.utils.error import CacheError


def _make_cache(config=None) -> object:
    cache_obj = Cache()
    cache_obj.init(
        data_manager=get_data_manager(data_path=f"data_map_{uuid.uuid4().hex}.txt"),
        config=config or Config(enable_token_counter=False),
    )
    return cache_obj


def test_memory_artifact_sqlite_round_trip(tmp_path) -> None:
    cache_a = _make_cache()
    cache_b = _make_cache()

    cache_a.record_intent(
        {"messages": [{"role": "user", "content": "Summarize Byte in one line"}]},
        session_id="memory-export",
    )
    remember_interaction(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Translate to French: Hello world"}],
        },
        answer="Bonjour le monde",
        reasoning="direct translation",
        cache_obj=cache_a,
    )

    artifact_path = tmp_path / "memory.sqlite"
    export_info = export_memory_artifact(str(artifact_path), cache_obj=cache_a, format="sqlite")
    import_info = import_memory_artifact(str(artifact_path), cache_obj=cache_b, format="sqlite")

    assert export_info["format"] == "sqlite"
    assert import_info["ai_memory"]["imported"] == 1
    assert cache_b.ai_memory_stats()["total_entries"] == 1
    assert cache_b.intent_stats()["total_records"] == 1


def test_memory_artifact_parquet_export_uses_parquet_writer(tmp_path, monkeypatch) -> None:
    pd = pytest.importorskip("pandas")
    cache_obj = _make_cache()
    remember_interaction(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Explain caching"}],
        },
        answer="Caching avoids repeat upstream calls.",
        cache_obj=cache_obj,
    )

    parquet_path = tmp_path / "memory.parquet"
    called = {"path": None}

    def fake_to_parquet(self, path, index=False) -> None:
        called["path"] = str(path)
        path.write_text(json.dumps(self.to_dict(orient="records")), encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)

    export_info = export_memory_artifact(str(parquet_path), cache_obj=cache_obj, format="parquet")

    assert export_info["format"] == "parquet"
    assert called["path"] == str(parquet_path)
    assert parquet_path.exists()


def test_memory_artifact_encryption_round_trip(tmp_path) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_encrypt_artifacts=True,
        security_encryption_key="artifact-secret",
    )
    cache_a = _make_cache(secure_config)
    cache_b = _make_cache(secure_config)

    remember_interaction(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Summarize patient intake notes"}],
        },
        answer="Patient intake summary",
        reasoning="brief summary",
        cache_obj=cache_a,
    )

    artifact_path = tmp_path / "memory.secure.json"
    export_info = export_memory_artifact(str(artifact_path), cache_obj=cache_a, format="json")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    import_info = import_memory_artifact(str(artifact_path), cache_obj=cache_b, format="json")

    assert export_info["format"] == "encrypted:json"
    assert payload["byte_secure_artifact"] is True
    assert import_info["ai_memory"]["imported"] == 1
    assert cache_b.ai_memory_stats()["total_entries"] == 1


def test_memory_artifact_zip_rejects_path_traversal(tmp_path) -> None:
    artifact_path = tmp_path / "memory.zip"
    with zipfile.ZipFile(artifact_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../snapshot.json", "{}")

    with pytest.raises(CacheError):
        load_snapshot_artifact(str(artifact_path), format="zip")
