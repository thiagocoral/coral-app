"""Testes do benchmark: estatística e execução contra o NAI falso."""
import json
import statistics

import httpx
import pytest
from fastapi.testclient import TestClient

from core.main import app
from core.modules import benchmark as bench
from core.modules import nai_client as nai
from tests import fake_nai

ENDPOINT = "http://nai.test/enterpriseai/gateway/v1/chat/completions"


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    def make_client(timeout=None, limits=None):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_nai.app), base_url="http://nai.test")

    monkeypatch.setattr(nai, "make_client", make_client)
    monkeypatch.setattr(nai, "DEFAULT_ENDPOINT", ENDPOINT)
    monkeypatch.setattr(nai, "DEFAULT_KEY", "good-key")
    monkeypatch.setattr(nai, "DEFAULT_MODEL", "timed-5-1-10")
    fake_nai.SEEN_PROMPTS.clear()
    fake_nai.TIMINGS.clear()
    fake_nai.STATS.update({"in_flight": 0, "max_in_flight": 0, "requests": 0})


client = TestClient(app)


def run(**body):
    resp = client.post("/api/v1/chat/benchmark", json=body)
    assert resp.status_code == 200
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def summary_of(events):
    return [e for e in events if e["type"] == "summary"][0]["summary"]


# --- estatística ---------------------------------------------------------

def test_percentile_linear_interpolation():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert bench.percentile(data, 50) == 5.5
    assert bench.percentile(data, 90) == pytest.approx(9.1)
    assert bench.percentile(data, 0) == 1 and bench.percentile(data, 100) == 10
    assert bench.percentile([7.0], 99) == 7.0
    assert bench.percentile([], 50) is None


def test_describe_matches_statistics_module():
    data = [12.5, 10.1, 11.7, 13.2, 9.8, 10.9]
    d = bench.describe(data + [None])
    assert d["n"] == 6
    assert d["mean"] == pytest.approx(statistics.mean(data))
    assert d["stdev"] == pytest.approx(statistics.stdev(data))
    assert d["p50"] == pytest.approx(statistics.median(data))
    assert d["min"] == 9.8 and d["max"] == 13.2
    assert bench.describe([None]) is None
    assert bench.describe([3.0])["stdev"] == 0.0


def test_payload_nonce_and_ignore_eos():
    p = bench.build_bench_payload("m", "olá", 64, 0.0, True, "abc")
    assert p["messages"] == [{"role": "user", "content": "[abc] olá"}]
    assert p["ignore_eos"] is True and p["max_tokens"] == 64 and p["temperature"] == 0.0
    assert "ignore_eos" not in bench.build_bench_payload("m", "olá", 64, 0.0, False, None)


# --- execução ------------------------------------------------------------

def test_benchmark_basic_flow():
    events = run(requests=5, concurrency=1, warmup=2, prompt="teste")
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    assert types.count("warmup") == 2 and types.count("result") == 5
    assert fake_nai.STATS["requests"] == 7  # 2 de aquecimento + 5 medidas
    s = summary_of(events)
    assert s["succeeded"] == 5 and s["failed"] == 0
    assert s["estimated_tokens"] is False
    assert s["metrics"]["output_tokens"]["mean"] == 10
    assert s["metrics"]["ttft_ms"]["n"] == 5 and s["metrics"]["decode_tps"]["mean"] > 0
    assert s["throughput_tps"] > 0 and s["settings"]["warmup"] == 2
    results = [e["result"] for e in events if e["type"] == "result"]
    assert sorted(r["index"] for r in results) == [1, 2, 3, 4, 5]


def test_avoid_prompt_cache_sends_unique_prompts():
    run(requests=4, concurrency=1, warmup=0, prompt="mesmo prompt")
    assert len(fake_nai.SEEN_PROMPTS) == 4
    fake_nai.SEEN_PROMPTS.clear()
    run(requests=4, concurrency=1, warmup=0, prompt="mesmo prompt", avoid_prompt_cache=False)
    assert fake_nai.SEEN_PROMPTS == {"mesmo prompt"}


def test_concurrency_is_respected():
    s = summary_of(run(model_name="timed-30-2-5", requests=8, concurrency=4, warmup=0))
    assert s["succeeded"] == 8
    assert fake_nai.STATS["max_in_flight"] == 4


def test_ignore_eos_and_max_tokens():
    s = summary_of(run(model_name="timed-5-1-10", requests=2, warmup=0, max_tokens=25, ignore_eos=True))
    assert s["metrics"]["output_tokens"]["mean"] == 25
    s = summary_of(run(model_name="timed-5-1-10", requests=2, warmup=0, max_tokens=4))
    assert s["metrics"]["output_tokens"]["mean"] == 4


def test_stream_options_fallback_marks_tokens_estimated():
    s = summary_of(run(model_name="no-usage", requests=2, warmup=1))
    assert s["succeeded"] == 2 and s["estimated_tokens"] is True and s["stream_usage"] is False


def test_endpoint_without_streaming():
    s = summary_of(run(model_name="json-only", requests=2, warmup=0))
    assert s["succeeded"] == 2 and s["streaming"] is False
    assert s["metrics"]["ttft_ms"] is None and s["metrics"]["e2e_ms"]["n"] == 2


def test_bad_key_stops_at_warmup():
    events = run(api_key="bad", requests=3, warmup=1)
    assert events[-1]["type"] == "error" and "401" in events[-1]["message"]
    assert not [e for e in events if e["type"] == "result"]


def test_bad_key_without_warmup_counts_failures():
    s = summary_of(run(api_key="bad", requests=3, warmup=0))
    assert s["succeeded"] == 0 and s["failed"] == 3 and "401" in s["errors"][0]
    assert s["metrics"]["ttft_ms"] is None


def test_validation_limits():
    assert client.post("/api/v1/chat/benchmark", json={"requests": 0}).status_code == 422
    assert client.post("/api/v1/chat/benchmark", json={"concurrency": 100}).status_code == 422
    assert client.post("/api/v1/chat/benchmark", json={"prompt": ""}).status_code == 422
