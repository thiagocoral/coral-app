"""Métricas do servidor de inferência (formato Prometheus, ex.: vLLM /metrics).

O uso do KV cache, a fila e o cache de prefixo não aparecem na API de chat:
ficam no endpoint /metrics do servidor de inferência (o pod do vLLM, por
exemplo). Quando a URL é informada, o benchmark coleta essas métricas a cada
0,5 s durante a execução (picos mais curtos que isso podem não aparecer).

Nomes aceitos (com ou sem o prefixo "vllm:", contadores com ou sem "_total"):
- kv_cache_usage_perc (vLLM atual) ou gpu_cache_usage_perc (nome antigo), fração 0-1
- num_requests_running, num_requests_waiting
- prefix_cache_hits, prefix_cache_queries (contadores, em tokens)
- num_preemptions, prompt_tokens, generation_tokens (contadores)
"""

import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core.modules import nai_client as nai

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+([^\s]+)(\s+\d+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

METRICS = {
    # nome lógico: (nomes-base em ordem de preferência, agregação)
    "kv_cache": (("kv_cache_usage_perc", "gpu_cache_usage_perc"), "max"),
    "running": (("num_requests_running",), "sum"),
    "waiting": (("num_requests_waiting",), "sum"),
    "prefix_hits": (("prefix_cache_hits",), "sum"),
    "prefix_queries": (("prefix_cache_queries",), "sum"),
    "preemptions": (("num_preemptions",), "sum"),
    "prompt_tokens": (("prompt_tokens",), "sum"),
    "generation_tokens": (("generation_tokens",), "sum"),
}

LABELS_PT = {
    "kv_cache": "Uso do KV cache",
    "running": "Requisições em execução",
    "waiting": "Requisições na fila",
    "prefix_hits": "Acertos no cache de prefixo",
    "prefix_queries": "Consultas ao cache de prefixo",
    "preemptions": "Preempções",
    "prompt_tokens": "Tokens de prompt processados",
    "generation_tokens": "Tokens gerados",
}


def parse_prometheus(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    series: Dict[str, List[Tuple[Dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, _, labels_str, value = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            number = float(value)
        except ValueError:
            continue
        labels = {k: v for k, v in _LABEL.findall(labels_str or "")}
        series.setdefault(name, []).append((labels, number))
    return series


def _candidates(base: str) -> List[str]:
    return [f"vllm:{base}", f"vllm:{base}_total", base, f"{base}_total"]


def extract(series: Dict[str, List[Tuple[Dict[str, str], float]]], model: Optional[str] = None) -> Dict[str, Any]:
    """Valores lógicos a partir das séries. Filtra pelo modelo quando o rótulo existe."""
    values: Dict[str, Any] = {}
    found: Dict[str, str] = {}
    for key, (bases, how) in METRICS.items():
        for base in bases:
            name = next((n for n in _candidates(base) if n in series), None)
            if not name:
                continue
            points = series[name]
            if model:
                same = [p for p in points if p[0].get("model_name") == model]
                if same:
                    points = same
            nums = [v for _, v in points]
            values[key] = max(nums) if how == "max" else sum(nums)
            found[key] = name
            break
    if "kv_cache" in values:
        kv = values["kv_cache"]
        values["kv_cache"] = kv * 100 if kv <= 1.5 else kv  # fração -> %
    values["_found"] = found
    return values


async def scrape(client: httpx.AsyncClient, url: str, model: Optional[str]) -> Dict[str, Any]:
    resp = await client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} em {url}")
    values = extract(parse_prometheus(resp.text), model)
    if not values["_found"]:
        raise RuntimeError("Nenhuma métrica conhecida (vLLM) encontrada nessa URL.")
    return values


async def check(url: str, model: Optional[str]) -> Dict[str, Any]:
    try:
        async with nai.make_client(httpx.Timeout(10.0, connect=nai.CONNECT_TIMEOUT)) as client:
            values = await scrape(client, url, model)
        found = values.pop("_found")
        return {"ok": True, "metrics": {LABELS_PT[k]: round(v, 3) for k, v in values.items()},
                "names": found}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)}


class Sampler:
    """Coleta as métricas a cada `interval` segundos em segundo plano."""

    def __init__(self, url: str, model: Optional[str], interval: float = 0.5):
        self.url = url
        self.model = model
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self.before: Optional[Dict[str, Any]] = None
        self.after: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.found: Dict[str, str] = {}
        self._task: Optional[asyncio.Task] = None
        self._t0 = 0.0
        self._client = nai.make_client(httpx.Timeout(5.0, connect=nai.CONNECT_TIMEOUT))

    async def _one(self) -> Optional[Dict[str, Any]]:
        try:
            values = await scrape(self._client, self.url, self.model)
            self.found = values.pop("_found")
            return values
        except Exception as e:  # noqa: BLE001
            self.error = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
            return None

    async def start(self, t0: float) -> None:
        self._t0 = t0
        self.before = await self._one()
        if self.before is not None:
            self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            t = time.perf_counter()
            values = await self._one()
            if values is not None:
                self.samples.append({"t": round(t - self._t0, 3), **{k: values.get(k) for k in
                                                                       ("kv_cache", "running", "waiting")}})
            await asyncio.sleep(max(0.0, self.interval - (time.perf_counter() - t)))

    async def stop(self) -> Dict[str, Any]:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self.before is not None:
            self.after = await self._one()
        await self._client.aclose()
        return self.report()

    def report(self) -> Dict[str, Any]:
        if self.before is None:
            return {"error": self.error or "Não foi possível ler as métricas.", "samples": []}

        def delta(key):
            if self.after is None or key not in self.after or key not in self.before:
                return None
            return self.after[key] - self.before[key]

        kv = [s["kv_cache"] for s in self.samples if s.get("kv_cache") is not None]
        hits, queries = delta("prefix_hits"), delta("prefix_queries")
        return {
            "error": self.error if not self.samples else None,
            "samples": self.samples,
            "found": self.found,
            "kv_peak": max(kv) if kv else None,
            "kv_mean": sum(kv) / len(kv) if kv else None,
            "running_peak": max((s["running"] for s in self.samples if s.get("running") is not None), default=None),
            "waiting_peak": max((s["waiting"] for s in self.samples if s.get("waiting") is not None), default=None),
            "prefix_hit_rate": (hits / queries * 100) if hits is not None and queries else None,
            "preemptions": delta("preemptions"),
            "server_prompt_tokens": delta("prompt_tokens"),
            "server_generation_tokens": delta("generation_tokens"),
        }
