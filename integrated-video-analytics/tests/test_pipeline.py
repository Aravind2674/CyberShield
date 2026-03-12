from pathlib import Path

import pipeline


def test_resolve_model_path_prefers_local_candidate(tmp_path):
    local_model = tmp_path / "best.pt"
    local_model.write_bytes(b"weights")

    resolved = pipeline.resolve_model_path(None, local_model, fallback="remote.pt")

    assert resolved == str(local_model)


def test_resolve_model_path_uses_fallback_when_missing(tmp_path):
    missing_model = tmp_path / "missing.pt"

    resolved = pipeline.resolve_model_path(None, missing_model, fallback="remote.pt")

    assert resolved == "remote.pt"


def test_normalize_plate_text_accepts_alphanumeric_plate():
    assert pipeline.normalize_plate_text("MH 12 AB 1234") == "MH12AB1234"


def test_normalize_plate_text_rejects_invalid_candidates():
    assert pipeline.normalize_plate_text("ABCDEFG") is None
    assert pipeline.normalize_plate_text("1234567") is None
    assert pipeline.normalize_plate_text("AB-1") is None


def test_read_env_float_uses_default_on_invalid_value(monkeypatch):
    monkeypatch.setenv("CYBERSHIELD_TEST_FLOAT", "O.30")

    assert pipeline.read_env_float("CYBERSHIELD_TEST_FLOAT", 0.3) == 0.3


def test_trim_timestamp_cache_expires_and_bounds_entries():
    cache = {
        "stale": 10.0,
        "fresh_a": 100.0,
        "fresh_b": 110.0,
        "fresh_c": 120.0,
    }

    pipeline.trim_timestamp_cache(cache, now=130.0, ttl_seconds=25.0, max_items=2)

    assert cache == {
        "fresh_b": 110.0,
        "fresh_c": 120.0,
    }
