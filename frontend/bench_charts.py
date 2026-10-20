"""Gráficos do benchmark (Altair). Funções puras: recebem dados, devolvem gráficos.

Convenções: "side" é o endpoint ("A" ou "B"); os tempos das requisições vêm do
backend em segundos relativos ao início da medição (start_s, first_s, end_s).
"""

from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

PHASE_PREFILL = "Fila + prefill"
PHASE_DECODE = "Geração"
PHASE_ERROR = "Erro"
PHASE_COLORS = alt.Scale(domain=[PHASE_PREFILL, PHASE_DECODE, PHASE_ERROR], range=["#f2a541", "#3b82c4", "#d1495b"])
_SIDE_PALETTE = {"Endpoint A": "#3b82c4", "Endpoint B": "#e07a1f"}


def side_scale(df: pd.DataFrame) -> alt.Scale:
    """Cores fixas por endpoint, mas só com os endpoints presentes (a legenda não mostra B à toa)."""
    present = [s for s in _SIDE_PALETTE if s in set(df["endpoint"])]
    return alt.Scale(domain=present, range=[_SIDE_PALETTE[s] for s in present])


PT_BR_LOCALE = {"number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["R$ ", ""]}}


def localized(chart):
    """Números no formato brasileiro (1.234,5) nos eixos e tooltips."""
    return chart.configure(locale=PT_BR_LOCALE)


def _side_label(side: str) -> str:
    return f"Endpoint {side}"


def _facet(chart: alt.Chart, sides: List[str]) -> alt.Chart:
    if len(sides) > 1:
        return chart.facet(row=alt.Row("endpoint:N", title=None)).resolve_scale(y="independent")
    return chart


# ---------------------------------------------------------------------------
# Rodada única
# ---------------------------------------------------------------------------

def gantt_rows(results_by_side: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for side, results in results_by_side.items():
        for r in results:
            if r.get("start_s") is None or r.get("end_s") is None:
                continue
            base = {"endpoint": _side_label(side), "req": r.get("index"),
                    "ttft_ms": r.get("ttft_ms"), "tokens": r.get("output_tokens")}
            if not r.get("ok"):
                rows.append({**base, "fase": PHASE_ERROR, "inicio": r["start_s"], "fim": r["end_s"]})
                continue
            first = r.get("first_s") if r.get("first_s") is not None else r["end_s"]
            rows.append({**base, "fase": PHASE_PREFILL, "inicio": r["start_s"], "fim": first})
            if first < r["end_s"]:
                rows.append({**base, "fase": PHASE_DECODE, "inicio": first, "fim": r["end_s"]})
    return pd.DataFrame(rows)


def gantt_chart(results_by_side: Dict[str, List[Dict[str, Any]]]) -> Optional[alt.Chart]:
    df = gantt_rows(results_by_side)
    if df.empty:
        return None
    n = int(df["req"].nunique())
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("inicio:Q", title="Tempo (s)"),
        x2="fim:Q",
        y=alt.Y("req:O", title="Requisição", axis=alt.Axis(labels=n <= 40, ticks=n <= 40),
                scale=alt.Scale(paddingInner=0.2 if n <= 80 else 0)),
        color=alt.Color("fase:N", scale=PHASE_COLORS, title=None, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("req:O", title="Requisição"), alt.Tooltip("fase:N"),
                 alt.Tooltip("inicio:Q", title="Início (s)", format=".3f"),
                 alt.Tooltip("fim:Q", title="Fim (s)", format=".3f"),
                 alt.Tooltip("ttft_ms:Q", title="1º token (ms)", format=".1f"),
                 alt.Tooltip("tokens:Q", title="Tokens")],
    ).properties(height=min(700, max(240, n * 12 + 90)))
    return _facet(chart, sorted(df["endpoint"].unique()))


def inflight_rows(results_by_side: Dict[str, List[Dict[str, Any]]], points: int = 300) -> pd.DataFrame:
    """Requisições em andamento ao longo do tempo, separadas por fase."""
    rows = []
    for side, results in results_by_side.items():
        spans = [r for r in results if r.get("start_s") is not None and r.get("end_s") is not None]
        if not spans:
            continue
        end = max(r["end_s"] for r in spans)
        step = end / points if end > 0 else 1
        for i in range(points + 1):
            t = i * step
            prefill = decode = 0
            for r in spans:
                first = r.get("first_s") if r.get("first_s") is not None else r["end_s"]
                if r["start_s"] <= t < first:
                    prefill += 1
                elif first <= t < r["end_s"]:
                    decode += 1
            rows.append({"endpoint": _side_label(side), "t": round(t, 4), "fase": PHASE_PREFILL, "n": prefill})
            rows.append({"endpoint": _side_label(side), "t": round(t, 4), "fase": PHASE_DECODE, "n": decode})
    return pd.DataFrame(rows)


def inflight_chart(results_by_side: Dict[str, List[Dict[str, Any]]]) -> Optional[alt.Chart]:
    df = inflight_rows(results_by_side)
    if df.empty:
        return None
    chart = alt.Chart(df).mark_area(interpolate="step-after", opacity=0.85).encode(
        x=alt.X("t:Q", title="Tempo (s)"),
        y=alt.Y("n:Q", stack=True, title="Requisições em andamento"),
        color=alt.Color("fase:N", scale=PHASE_COLORS, title=None, legend=alt.Legend(orient="top")),
        order=alt.Order("fase:N", sort="descending"),
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("t:Q", title="Tempo (s)", format=".2f"),
                 alt.Tooltip("fase:N"), alt.Tooltip("n:Q", title="Requisições")],
    ).properties(height=320)
    return _facet(chart, sorted(df["endpoint"].unique()))


def throughput_timeline_chart(summaries: Dict[str, Dict[str, Any]]) -> Optional[alt.Chart]:
    rows, steady = [], []
    for side, s in summaries.items():
        tl = (s or {}).get("timeline") or {}
        for t, v in zip(tl.get("t", []), tl.get("tokens_per_s", [])):
            rows.append({"endpoint": _side_label(side), "t": t, "tok_s": v})
        if s and s.get("steady_throughput_tps") and s.get("steady_window_s"):
            w0, w1 = s["steady_window_s"]
            steady.append({"endpoint": _side_label(side), "w0": w0, "w1": w1, "tok_s": s["steady_throughput_tps"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    line = alt.Chart(df).mark_line(interpolate="step-after").encode(
        x=alt.X("t:Q", title="Tempo (s)"),
        y=alt.Y("tok_s:Q", title="Tokens gerados por segundo (todas as requisições)"),
        color=alt.Color("endpoint:N", scale=side_scale(df), title=None, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("t:Q", title="Tempo (s)", format=".2f"),
                 alt.Tooltip("tok_s:Q", title="tok/s", format=",.1f")],
    )
    layers = [line]
    if steady:
        sd = pd.DataFrame(steady)
        layers.append(alt.Chart(sd).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
            x="w0:Q", x2="w1:Q", y="tok_s:Q", color=alt.Color("endpoint:N", scale=side_scale(df), legend=None),
            tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("tok_s:Q", title="Em regime (tok/s)", format=",.1f")]))
    return alt.layer(*layers).properties(height=340)


def server_rows(summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for side, s in summaries.items():
        server = (s or {}).get("server") or {}
        for sample in server.get("samples") or []:
            rows.append({"endpoint": _side_label(side), "t": sample["t"], "kv": sample.get("kv_cache"),
                         "running": sample.get("running"), "waiting": sample.get("waiting")})
    return pd.DataFrame(rows)


def kv_cache_chart(summaries: Dict[str, Dict[str, Any]]) -> Optional[alt.Chart]:
    df = server_rows(summaries)
    if df.empty or df["kv"].isna().all():
        return None
    df = df.dropna(subset=["kv"])
    return alt.Chart(df).mark_area(line=True, opacity=0.25, interpolate="monotone").encode(
        x=alt.X("t:Q", title="Tempo (s)"),
        y=alt.Y("kv:Q", title="Uso do KV cache (%)", scale=alt.Scale(domain=[0, max(100, float(df["kv"].max()))])),
        color=alt.Color("endpoint:N", scale=side_scale(df), title=None, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("t:Q", title="Tempo (s)", format=".1f"),
                 alt.Tooltip("kv:Q", title="KV cache (%)", format=".1f")],
    ).properties(height=320)


def server_queue_chart(summaries: Dict[str, Dict[str, Any]]) -> Optional[alt.Chart]:
    df = server_rows(summaries)
    if df.empty:
        return None
    long = df.melt(id_vars=["endpoint", "t"], value_vars=["running", "waiting"], var_name="estado", value_name="n")
    long = long.dropna(subset=["n"])
    if long.empty:
        return None
    long["estado"] = long["estado"].map({"running": "Em execução no servidor", "waiting": "Na fila do servidor"})
    chart = alt.Chart(long).mark_line(interpolate="step-after").encode(
        x=alt.X("t:Q", title="Tempo (s)"),
        y=alt.Y("n:Q", title="Requisições"),
        color=alt.Color("estado:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(range=["#3b82c4", "#d1495b"])),
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("t:Q", title="Tempo (s)", format=".1f"),
                 alt.Tooltip("estado:N"), alt.Tooltip("n:Q", title="Requisições")],
    ).properties(height=280)
    return _facet(chart, sorted(long["endpoint"].unique()))


def histogram_chart(results_by_side: Dict[str, List[Dict[str, Any]]], key: str, title: str) -> Optional[alt.Chart]:
    rows = [{"endpoint": _side_label(side), "v": r[key]}
            for side, results in results_by_side.items() for r in results
            if r.get("ok") and r.get(key) is not None]
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X("v:Q", bin=alt.Bin(maxbins=25), title=title),
        y=alt.Y("count():Q", stack=None, title="Requisições"),
        color=alt.Color("endpoint:N", scale=side_scale(df), title=None, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("count():Q", title="Requisições")],
    ).properties(height=280)


# ---------------------------------------------------------------------------
# Varreduras (vários níveis de concorrência ou de contexto)
# ---------------------------------------------------------------------------

def _metric(summary: Optional[Dict[str, Any]], key: str, stat: str) -> Optional[float]:
    d = ((summary or {}).get("metrics") or {}).get(key)
    return d.get(stat) if d else None


def sweep_rows(points_by_side: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for side, points in points_by_side.items():
        for p in points:
            s = p.get("summary")
            if not s or not s.get("succeeded"):
                continue
            server = s.get("server") or {}
            ttft_p50 = _metric(s, "ttft_ms", "p50")
            prompt_p50 = _metric(s, "prompt_tokens", "p50")
            rows.append({
                "endpoint": _side_label(side),
                "nivel": p["level"],
                "concorrencia": (s.get("settings") or {}).get("concurrency"),
                "prompt_tokens": prompt_p50,
                "ttft_p50": ttft_p50,
                "ttft_p90": _metric(s, "ttft_ms", "p90"),
                "e2e_p50": _metric(s, "e2e_ms", "p50"),
                "e2e_p90": _metric(s, "e2e_ms", "p90"),
                "decode_p50": _metric(s, "decode_tps", "p50"),
                "tpot_p50": _metric(s, "tpot_ms", "p50"),
                "throughput": s.get("steady_throughput_tps") or s.get("throughput_tps"),
                "throughput_total": s.get("throughput_tps"),
                "regime": bool(s.get("steady_throughput_tps")),
                "prefill_tps": (prompt_p50 / (ttft_p50 / 1000)) if prompt_p50 and ttft_p50 else None,
                "kv_peak": server.get("kv_peak"),
                "waiting_peak": server.get("waiting_peak"),
                "falhas": s.get("failed", 0),
            })
    return pd.DataFrame(rows)


def sweep_line(df: pd.DataFrame, x: str, x_title: str, series: List[tuple], y_title: str,
               log_x: bool = False) -> Optional[alt.Chart]:
    """series: [(coluna, rótulo)], desenhadas por endpoint (cor) e por série (traço)."""
    cols = [c for c, _ in series if c in df.columns and df[c].notna().any()]
    if df.empty or not cols:
        return None
    labels = dict(series)
    long = df.melt(id_vars=["endpoint", x], value_vars=cols, var_name="serie", value_name="y").dropna(subset=["y"])
    long["serie"] = long["serie"].map(labels)
    xscale = alt.Scale(type="log", base=2) if log_x else alt.Scale(zero=False)
    base = alt.Chart(long).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=xscale),
        y=alt.Y("y:Q", title=y_title),
        color=alt.Color("endpoint:N", scale=side_scale(long), title=None, legend=alt.Legend(orient="top")),
        detail="serie:N",
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip(f"{x}:Q", title=x_title, format=",.0f"),
                 alt.Tooltip("serie:N", title="Métrica"), alt.Tooltip("y:Q", title=y_title, format=",.1f")],
    )
    line = base.mark_line().encode(
        strokeDash=alt.StrokeDash("serie:N", title=None, sort=[labels[c] for c in cols],
                                  legend=alt.Legend(orient="top", symbolType="stroke") if len(cols) > 1 else None))
    return (line + base.mark_point(filled=True, size=50)).properties(height=320)


def latency_throughput_chart(df: pd.DataFrame, latency: str, latency_title: str) -> Optional[alt.Chart]:
    """Curva clássica de serving: quanto custa em latência cada ganho de throughput."""
    if df.empty or df["throughput"].isna().all() or df[latency].isna().all():
        return None
    d = df.dropna(subset=["throughput", latency]).sort_values(["endpoint", "concorrencia"])
    base = alt.Chart(d).encode(
        x=alt.X("throughput:Q", title="Throughput agregado (tok/s)"),
        y=alt.Y(f"{latency}:Q", title=latency_title),
        color=alt.Color("endpoint:N", scale=side_scale(d), title=None, legend=alt.Legend(orient="top")),
        order="concorrencia:Q",
        tooltip=[alt.Tooltip("endpoint:N"), alt.Tooltip("concorrencia:Q", title="Concorrência"),
                 alt.Tooltip("throughput:Q", title="tok/s", format=",.1f"),
                 alt.Tooltip(f"{latency}:Q", title=latency_title, format=",.1f")],
    )
    labels = base.transform_calculate(rotulo='"C=" + datum.concorrencia').mark_text(
        align="left", dx=7, dy=-7, fontSize=11).encode(text="rotulo:N")
    return (base.mark_line() + base.mark_point(filled=True, size=60) + labels).properties(height=400)
