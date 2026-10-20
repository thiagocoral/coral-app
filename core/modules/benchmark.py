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
                }, gaps

            t_first: Optional[float] = None
            t_last: Optional[float] = None
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
    }, gaps


def _failed(message: str, t_start: float) -> Dict[str, Any]:
    return {
        "ok": False, "error": message, "streaming": True,
        "ttft_ms": None, "e2e_ms": (time.perf_counter() - t_start) * 1000, "tpot_ms": None,
        "decode_tps": None, "output_tokens": None, "prompt_tokens": None,
        "estimated": False, "chunks": 0,
    }


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
) -> AsyncIterator[Dict[str, Any]]:
    """Gera eventos: start, warmup, result, summary, error, done."""
    concurrency = max(1, min(concurrency, requests))
    state = {"stream_options": True}

    def payload_for() -> Dict[str, Any]:
        nonce = uuid.uuid4().hex[:12] if avoid_prompt_cache else None
        return build_bench_payload(model, prompt, max_tokens, temperature, ignore_eos, nonce)

    async def one(client: httpx.AsyncClient) -> Tuple[Dict[str, Any], List[float]]:
        payload = payload_for()
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
    try:
        async with client:
            # Aquecimento: abre as conexões e aquece o modelo; não entra nas estatísticas
            warm_ok = 0
            for i in range(warmup):
                result, _ = await one(client)
                warm_ok += 1 if result["ok"] else 0
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
                    result["started_s"] = started
                    result["finished_s"] = time.perf_counter()
                    await queue.put((result, gaps))

            results: List[Dict[str, Any]] = []
            all_gaps: List[float] = []
            tasks = [asyncio.create_task(worker(i)) for i in range(requests)]
            for done in range(1, requests + 1):
                result, gaps = await queue.get()
                results.append(result)
                all_gaps.extend(gaps)
                public = {k: v for k, v in result.items() if k not in ("started_s", "finished_s")}
                yield {"type": "result", "done": done, "total": requests, "result": public}

            wall = (max(r["finished_s"] for r in results) - min(r["started_s"] for r in results)) if results else None
            summary = summarize(results, all_gaps, wall)
            summary["stream_usage"] = state["stream_options"]
            summary["settings"] = {
                "model": model, "requests": requests, "concurrency": concurrency, "warmup": warmup,
                "max_tokens": max_tokens, "temperature": temperature,
                "avoid_prompt_cache": avoid_prompt_cache, "ignore_eos": ignore_eos,
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
