"""NAI simulado (API compatível com OpenAI) para testes.

O comportamento muda conforme o nome do modelo:
- plain         : resposta normal, com usage no stream
- think         : usa <think>...</think> no content, com tags quebradas entre chunks
- r1            : só fecha </think> (template do DeepSeek-R1)
- reasoning     : usa delta.reasoning_content
- no-usage      : rejeita stream_options com HTTP 400
- json-only     : ignora stream e devolve JSON
- slow          : demora 1s antes do primeiro token
Qualquer modelo exige 'Authorization: Bearer good-key'.
"""
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
MODELS = ["plain", "think", "r1", "reasoning", "no-usage", "json-only", "slow"]
LAST = {}


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


@app.post("/enterpriseai/gateway/v1/chat/completions")
async def chat(request: Request):
    if not check_auth(request):
        return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401)
    body = await request.json()
    LAST.clear()
    LAST.update(body)
    model = body["model"]
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
