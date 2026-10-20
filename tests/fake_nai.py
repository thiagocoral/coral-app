"""NAI simulado (API compatível com OpenAI) para testes.

O comportamento muda conforme o nome do modelo:
- plain         : resposta normal, com usage no stream
- think         : usa <think>...</think> no content, com tags quebradas entre chunks
- r1            : só fecha </think> (template do DeepSeek-R1)
- reasoning     : usa delta.reasoning_content
- no-usage      : rejeita stream_options com HTTP 400
- json-only     : ignora stream e devolve JSON
- slow          : demora 1s antes do primeiro token
- timed-T-I-N   : primeiro token após T ms, um token a cada I ms, N tokens
                  (respeita max_tokens; com ignore_eos gera sempre max_tokens).
                  Prompt já visto responde com TTFT 10x menor (simula cache de prefixo).
- load-T-I-N    : como um servidor sob carga: TTFT = T ms + 0,02 ms por token de prompt,
                  intervalo = I ms × (1 + 0,15 × (em execução − 1)), lote máximo de 8
                  (o resto espera na fila), prompt_tokens ≈ caracteres ÷ 4 + 12.
                  Expõe /metrics no formato do vLLM (KV cache, fila, cache de prefixo).
Qualquer modelo exige 'Authorization: Bearer good-key'.
"""
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
MODELS = ["plain", "think", "r1", "reasoning", "no-usage", "json-only", "slow", "timed-200-20-50"]
LAST = {}
SEEN_PROMPTS = set()
STATS = {"in_flight": 0, "max_in_flight": 0, "requests": 0}
TIMINGS = []  # horários reais de envio no servidor (referência para validar a medição)
KV_CAPACITY = 20000  # tokens que cabem no KV cache simulado
LOAD = {"running": 0, "waiting": 0, "kv_tokens": 0, "kv_peak": 0.0, "prefix_hits": 0, "prefix_queries": 0,
        "prompt_tokens": 0, "generation_tokens": 0}
BATCHES = {}  # semáforo do lote por event loop (os testes usam loops diferentes)


def check_auth(request: Request):
    return request.headers.get("authorization") == "Bearer good-key"


@app.get("/enterpriseai/gateway/v1/models")
async def models(request: Request):
    if not check_auth(request):
        return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401)
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]}


@app.get("/last")
async def last():
    return LAST


@app.get("/stats")
async def stats():
    return STATS


@app.post("/reset")
async def reset():
    SEEN_PROMPTS.clear()
    TIMINGS.clear()
    LOAD.update({"running": 0, "waiting": 0, "kv_tokens": 0, "kv_peak": 0.0, "prefix_hits": 0,
                 "prefix_queries": 0, "prompt_tokens": 0, "generation_tokens": 0})
    STATS.update({"in_flight": 0, "max_in_flight": 0, "requests": 0})
    return STATS


@app.get("/timings")
async def timings():
    return TIMINGS


@app.get("/load")
async def load():
    return LOAD


@app.get("/metrics")
async def metrics():
    from fastapi.responses import PlainTextResponse
    model = 'model_name="load-model"'
    text = "\n".join([
        "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f"vllm:kv_cache_usage_perc{{{model},engine=\"0\"}} {LOAD['kv_tokens'] / KV_CAPACITY}",
        f"vllm:num_requests_running{{{model},engine=\"0\"}} {LOAD['running']}",
        f"vllm:num_requests_waiting{{{model},engine=\"0\"}} {LOAD['waiting']}",
        "# TYPE vllm:prefix_cache_hits counter",
        f"vllm:prefix_cache_hits_total{{{model}}} {LOAD['prefix_hits']}",
        f"vllm:prefix_cache_queries_total{{{model}}} {LOAD['prefix_queries']}",
        f"vllm:num_preemptions_total{{{model}}} 0",
        f"vllm:prompt_tokens_total{{{model}}} {LOAD['prompt_tokens']}",
        f"vllm:generation_tokens_total{{{model}}} {LOAD['generation_tokens']}",
        "",
    ])
    return PlainTextResponse(text)


async def load_stream(body, ttft_ms, itl_ms, n_tokens, t_received):
    batch = BATCHES.setdefault(id(asyncio.get_running_loop()), asyncio.Semaphore(8))
    prompt = body["messages"][-1]["content"]
    p_tokens = len(prompt) // 4 + 12
    n = body.get("max_tokens") if body.get("ignore_eos") else min(n_tokens, body.get("max_tokens") or n_tokens)
    LOAD["prefix_queries"] += p_tokens
    if prompt in SEEN_PROMPTS:
        LOAD["prefix_hits"] += p_tokens
    SEEN_PROMPTS.add(prompt)
    STATS["requests"] += 1
    LOAD["waiting"] += 1
    async with batch:
        LOAD["waiting"] -= 1
        LOAD["running"] += 1
        STATS["in_flight"] += 1
        STATS["max_in_flight"] = max(STATS["max_in_flight"], STATS["in_flight"])
        held = 0
        try:
            LOAD["kv_tokens"] += p_tokens
            held += p_tokens
            LOAD["prompt_tokens"] += p_tokens
            await asyncio.sleep((ttft_ms + 0.02 * p_tokens) / 1000)
            for i in range(n):
                if i:
                    await asyncio.sleep(itl_ms * (1 + 0.15 * (LOAD["running"] - 1)) / 1000)
                LOAD["kv_tokens"] += 1
                held += 1
                LOAD["generation_tokens"] += 1
                LOAD["kv_peak"] = max(LOAD["kv_peak"], LOAD["kv_tokens"] / KV_CAPACITY * 100)
                yield "data: " + json.dumps({"choices": [{"delta": {"content": f"t{i} "}}]}) + "\n\n"
            if body.get("stream_options", {}).get("include_usage"):
                yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": p_tokens, "completion_tokens": n,
                                                                      "total_tokens": p_tokens + n}}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            LOAD["kv_tokens"] -= held
            LOAD["running"] -= 1
            STATS["in_flight"] -= 1


async def timed_stream(body, ttft_ms, itl_ms, n_tokens, t_received):
    prompt = body["messages"][-1]["content"]
    cached = prompt in SEEN_PROMPTS
    SEEN_PROMPTS.add(prompt)
    n = body.get("max_tokens") if body.get("ignore_eos") else min(n_tokens, body.get("max_tokens") or n_tokens)
    STATS["requests"] += 1
    STATS["in_flight"] += 1
    STATS["max_in_flight"] = max(STATS["max_in_flight"], STATS["in_flight"])
    sent = []
    try:
        await asyncio.sleep((ttft_ms / 10 if cached else ttft_ms) / 1000)
        for i in range(n):
            if i:
                await asyncio.sleep(itl_ms / 1000)
            sent.append(time.perf_counter())
            yield "data: " + json.dumps({"choices": [{"delta": {"content": f"t{i} "}}]}) + "\n\n"
        TIMINGS.append({
            "ttft_ms": (sent[0] - t_received) * 1000,
            "tpot_ms": (sent[-1] - sent[0]) * 1000 / (len(sent) - 1) if len(sent) > 1 else None,
            "tokens": len(sent),
        })
        if body.get("stream_options", {}).get("include_usage"):
            yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": n, "total_tokens": 30 + n}}) + "\n\n"
        yield "data: [DONE]\n\n"
    finally:
        STATS["in_flight"] -= 1


@app.post("/enterpriseai/gateway/v1/chat/completions")
async def chat(request: Request):
    t_received = time.perf_counter()
    if not check_auth(request):
        return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401)
    body = await request.json()
    LAST.clear()
    LAST.update(body)
    model = body["model"]
    if model.startswith("load-"):
        _, t, i, n = model.split("-")
        return StreamingResponse(load_stream(body, float(t), float(i), int(n), t_received), media_type="text/event-stream")
    if model.startswith("timed-"):
        _, t, i, n = model.split("-")
        return StreamingResponse(timed_stream(body, float(t), float(i), int(n), t_received), media_type="text/event-stream")
    msgs = body["messages"]
    user = msgs[-1]["content"]
    n_hist = len([m for m in msgs if m["role"] != "system"]) - 1
    answer = f"Resposta para '{user}' (historico={n_hist}, temp={body.get('temperature')}, max={body.get('max_tokens')})"

    if model == "no-usage" and "stream_options" in body:
        return JSONResponse({"error": {"message": "stream_options not supported"}}, status_code=400)

    if model == "think":
        pieces = ["<th", "ink>Pensando ", "no problema", "</thi", "nk>\n\n"] + [w + " " for w in answer.split()]
    elif model == "r1":
        pieces = ["Raciocínio ", "inicial ", "</think>", "\n"] + [w + " " for w in answer.split()]
    else:
        pieces = [w + " " for w in answer.split()]

    if model == "json-only" or not body.get("stream"):
        content = "".join(pieces)
        msg = {"role": "assistant", "content": content}
        if model == "reasoning":
            msg["reasoning_content"] = "Raciocínio separado."
        return {"choices": [{"message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(pieces), "total_tokens": 10 + len(pieces)}}

    async def gen():
        if model == "slow":
            await asyncio.sleep(1.0)
        if model == "reasoning":
            for r in ["Raciocínio ", "separado."]:
                yield "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": r}}]}) + "\n\n"
                await asyncio.sleep(0.01)
        yield "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}) + "\n\n"
        for p in pieces:
            yield "data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) + "\n\n"
            await asyncio.sleep(0.02)
        yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        if body.get("stream_options", {}).get("include_usage"):
            yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": len(pieces), "total_tokens": 42 + len(pieces)}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
