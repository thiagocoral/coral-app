"""Os gráficos do benchmark são montados e validados contra o schema do Vega-Lite."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "frontend"))
import bench_charts as bc  # noqa: E402

RESULTS = {
    "A": [
        {"index": 1, "ok": True, "start_s": 0.0, "first_s": 0.05, "end_s": 1.0, "ttft_ms": 50, "e2e_ms": 1000,
         "decode_tps": 40.0, "output_tokens": 40},
        {"index": 2, "ok": True, "start_s": 0.1, "first_s": 0.2, "end_s": 1.2, "ttft_ms": 100, "e2e_ms": 1100,
         "decode_tps": 39.0, "output_tokens": 40},
        {"index": 3, "ok": False, "start_s": 0.3, "first_s": None, "end_s": 0.4, "error": "x"},
    ],
    "B": [
        {"index": 1, "ok": True, "start_s": 0.0, "first_s": 0.02, "end_s": 0.5, "ttft_ms": 20, "e2e_ms": 500,
         "decode_tps": 80.0, "output_tokens": 40},
    ],
}


def summary(**extra):
    base = {
        "succeeded": 2, "failed": 0, "throughput_tps": 70.0, "steady_throughput_tps": 72.0,
        "steady_window_s": [0.1, 0.8], "settings": {"concurrency": 2},
        "timeline": {"bin_s": 0.1, "t": [0, 0.1, 0.2], "tokens_per_s": [0, 50, 60]},
        "server": {"samples": [{"t": 0.0, "kv_cache": 1.5, "running": 1, "waiting": 0},
                               {"t": 0.5, "kv_cache": 3.0, "running": 2, "waiting": 1}],
                   "kv_peak": 3.0, "waiting_peak": 1},
        "metrics": {k: {"p50": v, "p90": v * 1.2} for k, v in
                    {"ttft_ms": 50, "e2e_ms": 900, "decode_tps": 40, "tpot_ms": 25, "prompt_tokens": 512}.items()},
    }
    base.update(extra)
    return base


def valid(chart):
    assert chart is not None
    spec = bc.localized(chart).to_dict()  # valida contra o schema do Vega-Lite
    assert spec["config"]["locale"]["number"]["decimal"] == ","
    return spec


def test_gantt_phases_and_errors():
    df = bc.gantt_rows(RESULTS)
    assert set(df["fase"]) == {bc.PHASE_PREFILL, bc.PHASE_DECODE, bc.PHASE_ERROR}
    assert len(df[df["endpoint"] == "Endpoint A"]) == 5  # 2 x (prefill + geração) + 1 erro
    valid(bc.gantt_chart(RESULTS))
    valid(bc.gantt_chart({"A": RESULTS["A"]}))


def test_inflight_counts():
    df = bc.inflight_rows({"A": RESULTS["A"]}, points=100)
    nearest = df.loc[(df["t"] - 0.15).abs().idxmin(), "t"]
    at = df[df["t"] == nearest]
    prefill = int(at[at["fase"] == bc.PHASE_PREFILL]["n"].iloc[0])
    decode = int(at[at["fase"] == bc.PHASE_DECODE]["n"].iloc[0])
    assert (prefill, decode) == (1, 1)  # req 2 esperando o 1º token, req 1 gerando
    valid(bc.inflight_chart(RESULTS))


def test_time_series_charts():
    summaries = {"A": summary(), "B": summary(server=None, steady_throughput_tps=None)}
    valid(bc.throughput_timeline_chart(summaries))
    valid(bc.kv_cache_chart(summaries))
    valid(bc.server_queue_chart(summaries))
    assert bc.kv_cache_chart({"A": summary(server=None)}) is None


def test_legend_only_lists_present_endpoints():
    spec = valid(bc.kv_cache_chart({"A": summary()}))
    assert spec["encoding"]["color"]["scale"]["domain"] == ["Endpoint A"]


def test_histograms():
    valid(bc.histogram_chart(RESULTS, "ttft_ms", "TTFT"))
    assert bc.histogram_chart({"A": RESULTS["A"][:1]}, "ttft_ms", "TTFT") is None


def test_sweep_charts():
    points = {
        "A": [{"level": c, "summary": summary(settings={"concurrency": c}, throughput_tps=50.0 * c,
                                               steady_throughput_tps=None)} for c in (1, 2, 4)],
        "B": [{"level": 1, "summary": None, "error": "falhou"},
              {"level": 2, "summary": summary(settings={"concurrency": 2})}],
    }
    df = bc.sweep_rows(points)
    assert len(df) == 4 and list(df[df["endpoint"] == "Endpoint A"]["throughput"]) == [50.0, 100.0, 200.0]
    assert df["prefill_tps"].iloc[0] == pytest.approx(512 / 0.05)
    valid(bc.sweep_line(df, "concorrencia", "C", [("ttft_p50", "p50"), ("ttft_p90", "p90")], "ms", log_x=True))
    valid(bc.sweep_line(df, "prompt_tokens", "tokens", [("kv_peak", "pico")], "%"))
    valid(bc.latency_throughput_chart(df, "ttft_p50", "TTFT"))
    assert bc.sweep_line(df.iloc[0:0], "concorrencia", "C", [("ttft_p50", "p50")], "ms") is None
