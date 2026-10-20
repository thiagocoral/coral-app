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
    monkeypatch.setattr(nai, "PROFILES", nai.load_profiles({"NAI_ENDPOINT": ENDPOINT, "NAI_API_KEY": "good-key"}))
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


# --- métricas do servidor (formato Prometheus) ----------------------------

from core.modules import server_metrics as sm  # noqa: E402

PROM = """
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="llama"} 0.42
vllm:kv_cache_usage_perc{engine="0",model_name="outro"} 0.90
vllm:num_requests_running{engine="0",model_name="llama"} 3.0
vllm:num_requests_running{engine="1",model_name="llama"} 2.0
vllm:num_requests_waiting{model_name="llama"} 1
vllm:prefix_cache_hits_total{model_name="llama"} 120.0
vllm:prefix_cache_queries_total{model_name="llama"} 480.0
vllm:generation_tokens_total{model_name="llama",label_with_quote="a\\"b"} 1000 1712345678
"""


def test_parse_and_extract_vllm_metrics():
    series = sm.parse_prometheus(PROM)
    v = sm.extract(series, "llama")
    assert v["kv_cache"] == pytest.approx(42.0)          # fração -> %, só do modelo pedido
    assert v["running"] == 5 and v["waiting"] == 1      # soma entre engines
    assert v["prefix_hits"] == 120 and v["generation_tokens"] == 1000
    assert v["_found"]["prefix_hits"] == "vllm:prefix_cache_hits_total"
    # sem filtro de modelo: KV usa o maior valor
    assert sm.extract(series)["kv_cache"] == pytest.approx(90.0)


def test_extract_legacy_and_unprefixed_names():
    v = sm.extract(sm.parse_prometheus("gpu_cache_usage_perc 0.5\nnum_requests_running 4\n"))
    assert v["kv_cache"] == pytest.approx(50.0) and v["running"] == 4
    v = sm.extract(sm.parse_prometheus("vllm:gpu_cache_usage_perc 0.25\n"))
    assert v["kv_cache"] == pytest.approx(25.0)
    assert sm.extract(sm.parse_prometheus("outra_metrica 1\n"))["_found"] == {}


def test_metrics_check_route():
    ok = client.post("/api/v1/chat/metrics-check", json={"metrics_url": "http://nai.test/metrics"}).json()
    assert ok["ok"] and ok["names"]["kv_cache"] == "vllm:kv_cache_usage_perc"
    bad = client.post("/api/v1/chat/metrics-check", json={"metrics_url": "http://nai.test/enterpriseai/gateway/v1/models"}).json()
    assert not bad["ok"]


# --- linha do tempo, regime estável, contexto e servidor --------------------

def test_timeline_sums_to_output_tokens():
    events = run(model_name="load-5-2-30", requests=6, concurrency=3, warmup=0)
    s = summary_of(events)
    tl = s["timeline"]
    assert tl["bin_s"] and len(tl["t"]) == len(tl["tokens_per_s"])
    total_from_timeline = sum(v * tl["bin_s"] for v in tl["tokens_per_s"])
    assert total_from_timeline == pytest.approx(s["total_output_tokens"], rel=1e-6)
    results = [e["result"] for e in events if e["type"] == "result"]
    for r in results:
        assert 0 <= r["start_s"] <= r["first_s"] <= r["end_s"]


def test_steady_state_window():
    s = summary_of(run(model_name="load-5-2-20", requests=12, concurrency=3, warmup=0))
    assert s["steady_throughput_tps"] and s["steady_throughput_tps"] > 0
    w0, w1 = s["steady_window_s"]
    assert w1 > w0
    s1 = summary_of(run(model_name="load-5-2-20", requests=3, concurrency=1, warmup=0))
    assert s1["steady_throughput_tps"] is None


def test_context_target_is_calibrated():
    events = run(model_name="load-5-1-5", requests=2, warmup=0, max_tokens=5, prompt_tokens_target=1500)
    cal = [e for e in events if e["type"] == "calibration"][0]
    assert cal["calibrated"] and cal["chars_per_token"] == pytest.approx(4.0, rel=0.01)
    s = summary_of(events)
    assert abs(s["metrics"]["prompt_tokens"]["p50"] - 1500) <= 3
    assert s["context"]["target"] == 1500


def test_server_metrics_collected_during_run():
    fake_nai.LOAD.update({"prompt_tokens": 0, "generation_tokens": 0, "kv_tokens": 0, "kv_peak": 0.0})
    s = summary_of(run(model_name="load-5-20-40", requests=8, concurrency=4, warmup=1,
                       metrics_url="http://nai.test/metrics"))
    server = s["server"]
    assert server["error"] is None and server["samples"]
    assert server["kv_peak"] > 0 and server["running_peak"] >= 1
    # o servidor contou exatamente os mesmos tokens que o cliente (aquecimento fora)
    assert server["server_generation_tokens"] == s["total_output_tokens"]
    assert server["server_prompt_tokens"] == s["total_prompt_tokens"]


def test_bad_metrics_url_does_not_break_benchmark():
    s = summary_of(run(model_name="load-5-1-5", requests=2, warmup=0, metrics_url="http://nai.test/nao-existe"))
    assert s["succeeded"] == 2 and s["server"]["error"]
