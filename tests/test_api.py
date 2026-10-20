"""Testes das rotas do backend contra um NAI falso (tests/fake_nai.py), sem rede."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from core.main import app
from core.modules import nai_client as nai
from tests import fake_nai

ENDPOINT = "http://nai.test/enterpriseai/gateway/v1/chat/completions"


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    def make_client(timeout=None):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_nai.app), base_url="http://nai.test")

    monkeypatch.setattr(nai, "make_client", make_client)
    monkeypatch.setattr(nai, "DEFAULT_ENDPOINT", ENDPOINT)
    monkeypatch.setattr(nai, "DEFAULT_KEY", "good-key")
    monkeypatch.setattr(nai, "DEFAULT_MODEL", "plain")


client = TestClient(app)


def stream(body):
    resp = client.post("/api/v1/chat/stream", json=body)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    reasoning, content = "", ""
    for e in events:
        if e["type"] == "reasoning":
            reasoning += e["content"]
        elif e["type"] == "content":
            content += e["content"]
        elif e["type"] == "reclassify":
            reasoning, content = reasoning + content, ""
    return events, reasoning.strip(), content.strip()


def test_health_and_root():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").json()["message"] == "Coral App Online"


def test_defaults_hide_api_key():
    data = client.get("/api/v1/chat/defaults").json()
    assert data["has_api_key"] is True
    assert "good-key" not in json.dumps(data)


@pytest.mark.parametrize("model", ["plain", "think", "r1", "reasoning", "no-usage", "json-only"])
def test_stream_variants(model):
    events, reasoning, content = stream({
        "user_input": "oi", "model_name": model,
        "history": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    })
    assert events[-1]["type"] == "done"
    assert content.startswith("Resposta para 'oi' (historico=2")
    metrics = [e for e in events if e["type"] == "metrics"][0]["metrics"]
    assert metrics["completion_tokens"] > 0
    assert metrics["estimated"] is (model == "no-usage")
    if model in ("think", "r1", "reasoning"):
        assert reasoning
    else:
        assert reasoning == ""


def test_stream_sends_parameters():
    _, _, content = stream({"user_input": "x", "temperature": 1.2, "max_tokens": 77, "system_prompt": "S"})
    assert "temp=1.2" in content and "max=77" in content
    assert fake_nai.LAST["messages"][0] == {"role": "system", "content": "S"}


def test_stream_bad_key_returns_error_event():
    events, _, _ = stream({"user_input": "oi", "api_key": "bad"})
    errors = [e for e in events if e["type"] == "error"]
    assert errors and "401" in errors[0]["message"] and "API Key" in errors[0]["message"]


def test_stream_document_notice(monkeypatch):
    monkeypatch.setattr(nai, "DOC_MAX_CHARS", 5)
    events, _, _ = stream({"user_input": "oi", "document_text": "abcdefghij", "document_name": "a.txt"})
    assert events[0]["type"] == "notice"
    assert "abcde" in fake_nai.LAST["messages"][0]["content"]


def test_ask_compat():
    data = client.post("/api/v1/chat/ask", json={"user_input": "oi", "model_name": "reasoning"}).json()
    assert data["status"] == "success"
    assert data["response"].startswith("Resposta para 'oi'")
    assert data["reasoning"] == "Raciocínio separado."
    assert data["metrics"]["total_tokens"] > 0


def test_models_and_connectivity():
    data = client.post("/api/v1/chat/models", json={}).json()
    assert data["status"] == "success" and "think" in data["models"]
    bad = client.post("/api/v1/chat/models", json={"api_key": "bad"}).json()
    assert bad["status"] == "error" and "401" in bad["message"]
    ok = client.post("/api/v1/chat/connectivity", json={}).json()
    assert ok["reachable"] and ok["ok"]
    auth = client.post("/api/v1/chat/connectivity", json={"api_key": "bad"}).json()
    assert auth["reachable"] and not auth["ok"] and auth["status_code"] == 401


def test_unreachable_host(monkeypatch):
    def make_client(timeout=None):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(nai, "make_client", make_client)
    events, _, _ = stream({"user_input": "oi"})
    assert events[-1]["type"] == "error" and "conectar" in events[-1]["message"]
    conn = client.post("/api/v1/chat/connectivity", json={}).json()
    assert conn["reachable"] is False
