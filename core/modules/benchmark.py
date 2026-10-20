"""Benchmark de endpoints NAI.

Roda dentro do backend (perto do NAI) para que as medições não incluam o
navegador nem o Streamlit. Cuidados para a medição ser precisa:

- Relógio monotônico de alta resolução (time.perf_counter), tomado no instante
  em que cada chunk SSE chega.
- Uma única conexão HTTP reaproveitada (keep-alive) e requisições de
  aquecimento descartadas, para não medir handshake TLS nem modelo "frio".
- Prefixo único em cada prompt (opcional, ligado por padrão) para que o cache
  de prefixo do vLLM não deixe o tempo até o primeiro token artificialmente baixo.
- Contagem de tokens pelo campo "usage" do próprio NAI. Só quando o endpoint
  não informa, os tokens são estimados pelo número de chunks (e marcados assim).

Métricas por requisição:
- TTFT: do envio até o primeiro chunk com texto (raciocínio ou resposta).
- E2E: do envio até o fim do stream.
- TPOT: (último token - primeiro token) / (tokens de saída - 1).
- Decode tok/s: 1 / TPOT, a velocidade de geração depois do primeiro token.
- ITL: intervalo entre chunks consecutivos.
"""

import asyncio
import json
import math
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from core.modules import nai_client as nai
from core.modules import server_metrics

DEFAULT_PROMPT = (
    "Explique, em cerca de 300 palavras, o que é Kubernetes e por que ele é usado "
    "para rodar aplicações de IA em produção."
)


# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------

def percentile(sorted_values: List[float], q: float) -> Optional[float]:
    """Percentil com interpolação linear (mesmo método padrão do numpy)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def describe(values: List[Optional[float]]) -> Optional[Dict[str, Any]]:
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    n = len(data)
    mean = sum(data) / n
    stdev = math.sqrt(sum((v - mean) ** 2 for v in data) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "stdev": stdev,
        "cv": (stdev / mean) if mean else None,
        "min": data[0],
        "p50": percentile(data, 50),
        "p90": percentile(data, 90),
        "p99": percentile(data, 99),
        "max": data[-1],
    }


def summarize(results: List[Dict[str, Any]], all_gaps_ms: List[float], wall_s: Optional[float]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    total_out = sum(r["output_tokens"] or 0 for r in ok)
    total_prompt = sum(r["prompt_tokens"] or 0 for r in ok)
    summary = {
        "requests": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "errors": sorted({r["error"] for r in results if not r.get("ok") and r.get("error")})[:5],
        "estimated_tokens": any(r.get("estimated") for r in ok),
        "streaming": all(r.get("streaming", True) for r in ok) if ok else None,
        "wall_time_s": wall_s,
        "total_output_tokens": total_out,
        "total_prompt_tokens": total_prompt,
        "throughput_tps": (total_out / wall_s) if wall_s and total_out else None,
        "requests_per_s": (len(ok) / wall_s) if wall_s and ok else None,
        "metrics": {
            "ttft_ms": describe([r["ttft_ms"] for r in ok]),
            "e2e_ms": describe([r["e2e_ms"] for r in ok]),
            "tpot_ms": describe([r["tpot_ms"] for r in ok]),
            "decode_tps": describe([r["decode_tps"] for r in ok]),
            "itl_ms": describe(all_gaps_ms),
            "output_tokens": describe([r["output_tokens"] for r in ok]),
            "prompt_tokens": describe([r["prompt_tokens"] for r in ok]),
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Medição de uma requisição
# ---------------------------------------------------------------------------

class StreamOptionsRejected(Exception):
    """O endpoint recusou stream_options; repetir sem ele."""


def _has_text(delta: Dict[str, Any]) -> bool:
    for field in ("content", "reasoning_content", "reasoning"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            return True
    return False


async def measure_request(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
    use_stream_options: bool,
) -> Tuple[Dict[str, Any], List[float]]:
    """Faz uma requisição em streaming e retorna (resultado, intervalos entre chunks em ms)."""
    body = {**payload, "stream": True}
    if use_stream_options:
        body["stream_options"] = {"include_usage": True}
    # JSON serializado antes de começar a contar o tempo
    content = json.dumps(body).encode("utf-8")

    gaps: List[float] = []
    sent = {}

    async def trace(event: str, info: Dict[str, Any]) -> None:
        # Instante exato em que o corpo da requisição terminou de ir para a rede.
        # Exclui do TTFT o trabalho do próprio cliente (montar a requisição, pool).
        if event.endswith("send_request_body.complete"):
            sent["t"] = time.perf_counter()

    t_start = time.perf_counter()
    try:
        async with client.stream(
            "POST", endpoint, content=content, headers=nai.auth_headers(api_key), extensions={"trace": trace}
        ) as resp:
            t_start = sent.get("t", t_start)
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "replace")
                if use_stream_options and resp.status_code in (400, 422):
                    raise StreamOptionsRejected(text)
                return _failed(nai._error_text(resp.status_code, text), t_start), gaps

            if "text/event-stream" not in resp.headers.get("content-type", ""):
                # Endpoint ignorou o streaming: só dá para medir o tempo total
                data = json.loads(await resp.aread())
                t_end = time.perf_counter()
                usage = data.get("usage") or {}
                e2e = (t_end - t_start) * 1000
                out = usage.get("completion_tokens")
                return {
                    "ok": True, "error": None, "streaming": False,
                    "ttft_ms": None, "e2e_ms": e2e, "tpot_ms": None,
                    "decode_tps": (out / (e2e / 1000)) if out else None,
                    "output_tokens": out, "prompt_tokens": usage.get("prompt_tokens"),
                    "estimated": out is None, "chunks": 0,
                    "_t_send": t_start, "_t_first": None, "_t_end": t_end, "_chunk_times": [],
                }, gaps

            t_first: Optional[float] = None
            t_last: Optional[float] = None
            chunk_times: List[float] = []
            chunks = 0
            usage: Dict[str, Any] = {}
            async for line in resp.aiter_lines():
                now = time.perf_counter()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except ValueError:
                    continue
                if chunk.get("error"):
                    err = chunk["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    return _failed(f"Erro do NAI durante a geração: {msg}", t_start), gaps
                if chunk.get("usage"):
                    usage = chunk["usage"]
                if any(_has_text(c.get("delta") or {}) for c in chunk.get("choices") or []):
                    if t_first is None:
                        t_first = now
                    else:
                        gaps.append((now - t_last) * 1000)
                    t_last = now
                    chunk_times.append(now)
                    chunks += 1
            t_end = time.perf_counter()
    except StreamOptionsRejected:
        raise
    except httpx.TimeoutException:
        return _failed("Tempo esgotado esperando o NAI.", t_start), gaps
    except httpx.HTTPError as e:
        return _failed(f"Falha de conexão: {e}", t_start), gaps

    if t_first is None:
        return _failed("O endpoint não retornou nenhum token.", t_start), gaps

    out = usage.get("completion_tokens")
    estimated = out is None
    if estimated:
        out = chunks
    decode_s = t_last - t_first
    tpot_ms = (decode_s * 1000 / (out - 1)) if out and out > 1 and decode_s > 0 else None
    return {
        "ok": True, "error": None, "streaming": True,
        "ttft_ms": (t_first - t_start) * 1000,
        "e2e_ms": (t_end - t_start) * 1000,
        "tpot_ms": tpot_ms,
        "decode_tps": (1000.0 / tpot_ms) if tpot_ms else None,
        "output_tokens": out,
        "prompt_tokens": usage.get("prompt_tokens"),
        "estimated": estimated,
        "chunks": chunks,
        "_t_send": t_start, "_t_first": t_first, "_t_end": t_end, "_chunk_times": chunk_times,
    }, gaps


def _failed(message: str, t_start: float) -> Dict[str, Any]:
    t_end = time.perf_counter()
    return {
        "ok": False, "error": message, "streaming": True,
        "ttft_ms": None, "e2e_ms": (t_end - t_start) * 1000, "tpot_ms": None,
        "decode_tps": None, "output_tokens": None, "prompt_tokens": None,
        "estimated": False, "chunks": 0,
        "_t_send": t_start, "_t_first": None, "_t_end": t_end, "_chunk_times": [],
    }


# ---------------------------------------------------------------------------
# Linha do tempo e regime estável
# ---------------------------------------------------------------------------

def timeline(results: List[Dict[str, Any]], t0: float) -> Dict[str, Any]:
    """Tokens/s ao longo do tempo, em ~100 intervalos.

    Cada chunk vale (tokens de saída da requisição ÷ chunks), para que a soma bata
    com o total informado pelo NAI mesmo quando um chunk traz mais de um token.
    """
    points: List[Tuple[float, float]] = []
    for r in results:
        times = r.get("_chunk_times") or []
        if not times or not r.get("output_tokens"):
            continue
        weight = r["output_tokens"] / len(times)
        points.extend((t - t0, weight) for t in times)
    ends = [r["_t_end"] - t0 for r in results if r.get("_t_end")]
    if not points or not ends:
        return {"bin_s": None, "t": [], "tokens_per_s": []}
    total = max(ends)
    bin_s = max(0.1, round(total / 100, 2))
    n = int(math.ceil(total / bin_s)) or 1
    bins = [0.0] * n
    for t, w in points:
        idx = min(n - 1, max(0, int(t // bin_s)))
        bins[idx] += w
    return {"bin_s": bin_s, "t": [round(i * bin_s, 3) for i in range(n)],
            "tokens_per_s": [round(b / bin_s, 2) for b in bins]}


def steady_state(results: List[Dict[str, Any]], t0: float, concurrency: int) -> Dict[str, Any]:
    """Throughput no regime estável: do instante em que as N vagas ficam ocupadas
    até o início da última requisição (depois disso as vagas começam a esvaziar).
    Exclui a subida e a descida da carga, que puxam a média total para baixo."""
    starts = sorted(r["_t_send"] - t0 for r in results if r.get("_t_send"))
    if concurrency < 2 or len(starts) < 2 * concurrency:
        return {"steady_throughput_tps": None, "steady_window_s": None}
    w0, w1 = starts[concurrency - 1], starts[-1]
    # A janela precisa cobrir ao menos uma requisição típica inteira para medir ciclos completos
    durations = sorted(r["_t_end"] - r["_t_send"] for r in results if r.get("ok") and r.get("_t_end"))
    typical = durations[len(durations) // 2] if durations else 0
    if w1 - w0 <= 0 or w1 - w0 < typical:
        return {"steady_throughput_tps": None, "steady_window_s": None}
    tokens = 0.0
    for r in results:
        times = r.get("_chunk_times") or []
        if not times or not r.get("output_tokens"):
            continue
        weight = r["output_tokens"] / len(times)
        tokens += weight * sum(1 for t in times if w0 <= t - t0 <= w1)
    return {"steady_throughput_tps": tokens / (w1 - w0), "steady_window_s": [round(w0, 3), round(w1, 3)]}


# ---------------------------------------------------------------------------
# Prompt com tamanho de contexto definido
# ---------------------------------------------------------------------------

FILLER = (
    "O Nutanix Kubernetes Platform simplifica a operação de clusters Kubernetes em ambientes híbridos, "
    "com governança centralizada, observabilidade integrada e ciclo de vida automatizado. Aplicações de "
    "inteligência artificial se beneficiam de escalabilidade elástica, isolamento de cargas e acesso "
    "padronizado a GPUs, enquanto equipes de plataforma mantêm segurança, conformidade e custos sob controle. "
)


def context_prompt(question: str, filler_chars: int) -> str:
    reps = filler_chars // len(FILLER) + 1
    filler = (FILLER * reps)[:max(0, filler_chars)]
    return f"Contexto:\n{filler}\n\nCom base no contexto acima, responda: {question}"


# ---------------------------------------------------------------------------
# Execução do benchmark
# ---------------------------------------------------------------------------

def build_bench_payload(
    model: str, prompt: str, max_tokens: int, temperature: float, ignore_eos: bool, nonce: Optional[str]
) -> Dict[str, Any]:
    content = f"[{nonce}] {prompt}" if nonce else prompt
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if ignore_eos:
        payload["ignore_eos"] = True  # parâmetro do vLLM: gera sempre max_tokens
    return payload


async def run_benchmark(
    endpoint: str,
    api_key: Optional[str],
    model: str,
    prompt: str,
    requests: int,
    concurrency: int,
    warmup: int,
    max_tokens: int,
    temperature: float,
    avoid_prompt_cache: bool,
    ignore_eos: bool,
    metrics_url: Optional[str] = None,
    prompt_tokens_target: Optional[int] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Gera eventos: start, calibration, warmup, result, summary, error, done."""
    concurrency = max(1, min(concurrency, requests))
    state = {"stream_options": True, "prompt": prompt}

    def payload_for(text: Optional[str] = None, tokens: Optional[int] = None) -> Dict[str, Any]:
        nonce = uuid.uuid4().hex[:12] if avoid_prompt_cache else None
        return build_bench_payload(model, text if text is not None else state["prompt"],
                                   tokens or max_tokens, temperature, ignore_eos and not tokens, nonce)

    async def one(client: httpx.AsyncClient, payload: Optional[Dict[str, Any]] = None):
        payload = payload or payload_for()
        try:
            return await measure_request(client, endpoint, api_key, payload, state["stream_options"])
        except StreamOptionsRejected:
            state["stream_options"] = False
            return await measure_request(client, endpoint, api_key, payload, False)

    # Pool com uma conexão por requisição simultânea, reaproveitadas (keep-alive)
    limits = httpx.Limits(max_connections=concurrency + 2, max_keepalive_connections=concurrency + 2)
    timeout = httpx.Timeout(nai.REQUEST_TIMEOUT, connect=nai.CONNECT_TIMEOUT)
    client = nai.make_client(timeout, limits=limits)
    yield {"type": "start", "requests": requests, "concurrency": concurrency, "warmup": warmup, "model": model}

    tasks: List[asyncio.Task] = []
    sampler: Optional[server_metrics.Sampler] = None
    context_info: Optional[Dict[str, Any]] = None
    try:
        async with client:
            if prompt_tokens_target:
                # Calibração: mede quantos caracteres valem um token neste modelo (tokenizador real)
                r0, _ = await one(client, payload_for(context_prompt(prompt, 0), 1))
                if not r0["ok"]:
                    yield {"type": "error", "message": f"Falha na calibração do contexto: {r0['error']}"}
                    return
                probe = 4000
                r1, _ = await one(client, payload_for(context_prompt(prompt, probe), 1))
                p0, p1 = r0.get("prompt_tokens"), (r1.get("prompt_tokens") if r1["ok"] else None)
                if p0 and p1 and p1 > p0:
                    cpt, calibrated = probe / (p1 - p0), True
                else:
                    cpt, calibrated, p0 = 4.0, False, p0 or 30
                filler = max(0, int(round((prompt_tokens_target - p0) * cpt)))
                state["prompt"] = context_prompt(prompt, filler)
                context_info = {"target": prompt_tokens_target, "chars_per_token": round(cpt, 3),
                                "calibrated": calibrated, "base_tokens": p0}
                yield {"type": "calibration", **context_info}

            # Aquecimento: abre as conexões e aquece o modelo; não entra nas estatísticas
            for i in range(warmup):
                result, _ = await one(client)
                yield {"type": "warmup", "done": i + 1, "total": warmup, "ok": result["ok"], "error": result["error"]}
                if not result["ok"] and i == 0:
                    yield {"type": "error", "message": f"Falha no aquecimento: {result['error']}"}
                    return

            semaphore = asyncio.Semaphore(concurrency)
            queue: asyncio.Queue = asyncio.Queue()

            async def worker(index: int):
                async with semaphore:
                    started = time.perf_counter()
                    result, gaps = await one(client)
                    result["index"] = index + 1
                    result["_started"] = started
                    result["_finished"] = time.perf_counter()
                    await queue.put((result, gaps))

            t0 = time.perf_counter()
            if metrics_url:
                sampler = server_metrics.Sampler(metrics_url, model)
                await sampler.start(t0)

            results: List[Dict[str, Any]] = []
            all_gaps: List[float] = []
            tasks = [asyncio.create_task(worker(i)) for i in range(requests)]
            for done in range(1, requests + 1):
                result, gaps = await queue.get()
                results.append(result)
                all_gaps.extend(gaps)
                public = {k: v for k, v in result.items() if not k.startswith("_")}
                public["start_s"] = round(result["_t_send"] - t0, 4)
                public["first_s"] = round(result["_t_first"] - t0, 4) if result.get("_t_first") else None
                public["end_s"] = round(result["_t_end"] - t0, 4)
                yield {"type": "result", "done": done, "total": requests, "result": public}

            server = await sampler.stop() if sampler else None
            sampler = None
            wall = (max(r["_finished"] for r in results) - min(r["_started"] for r in results)) if results else None
            summary = summarize(results, all_gaps, wall)
            summary["stream_usage"] = state["stream_options"]
            summary["timeline"] = timeline(results, t0)
            summary.update(steady_state(results, t0, concurrency))
            summary["server"] = server
            summary["context"] = context_info
            summary["settings"] = {
                "model": model, "requests": requests, "concurrency": concurrency, "warmup": warmup,
                "max_tokens": max_tokens, "temperature": temperature,
                "avoid_prompt_cache": avoid_prompt_cache, "ignore_eos": ignore_eos,
                "prompt_tokens_target": prompt_tokens_target, "metrics_url": bool(metrics_url),
            }
            yield {"type": "summary", "summary": summary}
            yield {"type": "done"}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"Erro no benchmark: {type(e).__name__}: {e}"}
    finally:
        # Cliente desconectou ou houve erro: cancela o que ainda estiver rodando
        for task in tasks:
            if not task.done():
                task.cancel()
        if sampler is not None:
            await sampler.stop()
