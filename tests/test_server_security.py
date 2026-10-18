"""Dashboard server hardening: CSRF/Origin guard and cycle-id validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from server import app as server_app  # noqa: E402

client = TestClient(server_app.app)


def test_cross_origin_post_is_rejected():
    r = client.post("/api/shutdown", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_same_origin_post_passes_the_guard():
    # The guard lets the request through; the handler itself may still refuse
    # (no engine attached in tests), but never with the guard's 403.
    r = client.post("/api/pause", headers={"Origin": "http://testserver"})
    assert r.status_code != 403


def test_post_without_origin_header_is_allowed():
    r = client.post("/api/pause")
    assert r.status_code != 403


def test_cross_origin_websocket_is_rejected():
    with pytest.raises(Exception):
        with client.websocket_connect("/ws", headers={"Origin": "https://evil.example"}):
            pass


def test_cycle_log_rejects_unknown_ids_and_traversal():
    assert client.get("/api/cycles/does-not-exist/log").json() == {"lines": []}
    r = client.get("/api/cycles/..%2F..%2Fetc/log")
    assert r.status_code == 404 or r.json() == {"lines": []}
