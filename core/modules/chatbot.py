"""Rotas do chat com o NAI.

- POST /api/v1/chat/stream        -> resposta em streaming (NDJSON, um evento por linha)
- POST /api/v1/chat/ask           -> resposta completa (mantida por compatibilidade)
- POST /api/v1/chat/models        -> lista os modelos do endpoint
- POST /api/v1/chat/connectivity  -> testa rede, TLS e autenticação
- POST /api/v1/chat/benchmark     -> benchmark do endpoint (NDJSON com progresso)
- POST /api/v1/chat/metrics-check -> testa o /metrics do servidor de inferência (KV cache etc.)
- GET  /api/v1/chat/profiles      -> endpoints salvos no servidor (sem as API Keys)
- GET  /api/v1/chat/defaults      -> valores padrão configurados no container
"""

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.modules import benchmark as bench
from core.modules import nai_client as nai
from core.modules import server_metrics

router = APIRouter()


class ChatMessage(BaseModel):
    user_input: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=131072)
    document_text: Optional[str] = None
    document_name: Optional[str] = None
    profile: Optional[str] = None


class EndpointInfo(BaseModel):
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    profile: Optional[str] = None


class BenchmarkRequest(BaseModel):
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    prompt: str = Field(default=bench.DEFAULT_PROMPT, min_length=1)
    requests: int = Field(default=10, ge=1, le=500)
    concurrency: int = Field(default=1, ge=1, le=64)
    warmup: int = Field(default=1, ge=0, le=10)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    avoid_prompt_cache: bool = True
    ignore_eos: bool = False
    profile: Optional[str] = None
    metrics_url: Optional[str] = None
    prompt_tokens_target: Optional[int] = Field(default=None, ge=16, le=1_000_000)


class MetricsCheck(BaseModel):
    metrics_url: str = Field(min_length=1)
    model_name: Optional[str] = None


def _resolve(endpoint_url: Optional[str], api_key: Optional[str], model_name: Optional[str] = None,
             profile: Optional[str] = None):
    endpoint = (endpoint_url or "").strip() or nai.DEFAULT_ENDPOINT
    key = nai.resolve_api_key(endpoint, api_key, profile)
    model = (model_name or "").strip() or nai.DEFAULT_MODEL
    return endpoint, key, model


def _payload(message: ChatMessage):
    endpoint, key, model = _resolve(message.endpoint_url, message.api_key, message.model_name, message.profile)
    payload, truncated = nai.build_payload(
        user_input=message.user_input,
        history=message.history,
        model=model,
        system_prompt=message.system_prompt,
        temperature=message.temperature,
        max_tokens=message.max_tokens,
        document_text=message.document_text,
        document_name=message.document_name,
    )
    return endpoint, key, payload, truncated


@router.post("/stream")
async def stream_chatbot(message: ChatMessage):
    endpoint, key, payload, truncated = _payload(message)

    async def events():
        if truncated:
            yield json.dumps({
                "type": "notice",
                "message": f"O documento foi cortado nos primeiros {nai.DOC_MAX_CHARS:,} caracteres.",
            }) + "\n"
        async for event in nai.stream_chat(endpoint, key, payload):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    endpoint, key, payload, truncated = _payload(message)
    result = await nai.complete_chat(endpoint, key, payload)
    if truncated:
        result["notice"] = f"O documento foi cortado nos primeiros {nai.DOC_MAX_CHARS:,} caracteres."
    return result


@router.post("/models")
async def list_models(info: EndpointInfo):
    endpoint, key, _ = _resolve(info.endpoint_url, info.api_key, profile=info.profile)
    return await nai.list_models(endpoint, key)


@router.post("/connectivity")
async def connectivity(info: EndpointInfo):
    endpoint, key, _ = _resolve(info.endpoint_url, info.api_key, profile=info.profile)
    return await nai.check_connectivity(endpoint, key)


@router.post("/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    endpoint, key, model = _resolve(req.endpoint_url, req.api_key, req.model_name, req.profile)

    async def events():
        async for event in bench.run_benchmark(
            endpoint=endpoint, api_key=key, model=model, prompt=req.prompt,
            requests=req.requests, concurrency=req.concurrency, warmup=req.warmup,
            max_tokens=req.max_tokens, temperature=req.temperature,
            avoid_prompt_cache=req.avoid_prompt_cache, ignore_eos=req.ignore_eos,
            metrics_url=(req.metrics_url or "").strip() or None,
            prompt_tokens_target=req.prompt_tokens_target,
        ):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/metrics-check")
async def metrics_check(req: MetricsCheck):
    """Lê uma vez o /metrics do servidor de inferência e diz quais métricas existem."""
    return await server_metrics.check(req.metrics_url.strip(), (req.model_name or "").strip() or None)


@router.get("/profiles")
async def profiles():
    """Endpoints salvos por variáveis de ambiente. As API Keys não saem do servidor."""
    return {"profiles": nai.public_profiles()}


@router.get("/defaults")
async def defaults():
    """Valores padrão do backend (a API Key nunca é exposta)."""
    return {
        "endpoint_url": nai.DEFAULT_ENDPOINT,
        "model_name": nai.DEFAULT_MODEL,
        "system_prompt": nai.DEFAULT_SYSTEM_PROMPT,
        "temperature": nai.DEFAULT_TEMPERATURE,
        "max_tokens": nai.DEFAULT_MAX_TOKENS,
        "max_history_messages": nai.MAX_HISTORY_MESSAGES,
        "doc_max_chars": nai.DOC_MAX_CHARS,
        "has_api_key": bool(nai.DEFAULT_KEY),
        "benchmark_prompt": bench.DEFAULT_PROMPT,
    }
