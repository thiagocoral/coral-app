"""Rotas do chat com o NAI.

- POST /api/v1/chat/stream        -> resposta em streaming (NDJSON, um evento por linha)
- POST /api/v1/chat/ask           -> resposta completa (mantida por compatibilidade)
- POST /api/v1/chat/models        -> lista os modelos do endpoint
- POST /api/v1/chat/connectivity  -> testa rede, TLS e autenticação
- GET  /api/v1/chat/defaults      -> valores padrão configurados no container
"""

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.modules import nai_client as nai

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


class EndpointInfo(BaseModel):
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None


def _resolve(endpoint_url: Optional[str], api_key: Optional[str], model_name: Optional[str] = None):
    endpoint = (endpoint_url or "").strip() or nai.DEFAULT_ENDPOINT
    key = (api_key or "").strip() or nai.DEFAULT_KEY
    model = (model_name or "").strip() or nai.DEFAULT_MODEL
    return endpoint, key, model


def _payload(message: ChatMessage):
    endpoint, key, model = _resolve(message.endpoint_url, message.api_key, message.model_name)
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
    endpoint, key, _ = _resolve(info.endpoint_url, info.api_key)
    return await nai.list_models(endpoint, key)


@router.post("/connectivity")
async def connectivity(info: EndpointInfo):
    endpoint, key, _ = _resolve(info.endpoint_url, info.api_key)
    return await nai.check_connectivity(endpoint, key)


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
    }
