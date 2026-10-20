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
    STATS.update({"in_flight": 0, "max_in_flight": 0, "requests": 0})
    return STATS


@app.get("/timings")
async def timings():
    return TIMINGS


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
