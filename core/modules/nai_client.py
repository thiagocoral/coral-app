"""Cliente para endpoints NAI (API compatível com OpenAI).

Concentra tudo que fala com o NAI: montagem do payload, streaming via SSE,
separação do raciocínio (reasoning) do texto final, cálculo de métricas,
listagem de modelos e teste de conectividade.

Toda configuração vem de variáveis de ambiente opcionais, com valores padrão,
para que o manifesto atual continue funcionando sem alterações.
"""

import json
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

import httpx


# ---------------------------------------------------------------------------
# Configuração (variáveis de ambiente opcionais)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_ENDPOINT = os.getenv(
    "NAI_ENDPOINT", "https://10.42.163.114/enterpriseai/gateway/v1/chat/completions"
)
DEFAULT_KEY = os.getenv("NAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "unified-endpoint")
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Você é o assistente NTNX BR, rodando no Nutanix Enterprise AI (NAI). "
    "Responda de forma clara e objetiva, em português do Brasil, a menos que "
    "o usuário escreva em outro idioma.",
)
DEFAULT_TEMPERATURE = _env_float("DEFAULT_TEMPERATURE", 0.3)
DEFAULT_MAX_TOKENS = _env_int("DEFAULT_MAX_TOKENS", 2048)
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 20)
DOC_MAX_CHARS = _env_int("DOC_MAX_CHARS", 20000)
REQUEST_TIMEOUT = _env_float("NAI_REQUEST_TIMEOUT", 300.0)
CONNECT_TIMEOUT = _env_float("NAI_CONNECT_TIMEOUT", 10.0)


# ---------------------------------------------------------------------------
# Perfis de endpoint (endpoints salvos)
# ---------------------------------------------------------------------------
#
# Definidos por variáveis de ambiente no backend, por exemplo:
#   NAI_PROFILE_1_NAME=GPU L40S - Llama 3.1 8B
#   NAI_PROFILE_1_URL=https://nai.local/enterpriseai/gateway/v1/chat/completions
#   NAI_PROFILE_1_MODEL=llama-31-8b
#   NAI_PROFILE_1_API_KEY=...        (de preferência vindo de um Secret)
# O endpoint padrão (NAI_ENDPOINT / MODEL_NAME / NAI_API_KEY) vira o perfil "default".
# As API Keys ficam só no servidor: a rota /profiles nunca as devolve.

_PROFILE_URL = re.compile(r"^NAI_PROFILE_(\d+)_URL$")


def endpoint_host(url: Optional[str]) -> str:
    """scheme://host:porta normalizado, usado para decidir se uma chave pode ser enviada."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def load_profiles(env: Optional[Mapping[str, str]] = None) -> List[Dict[str, str]]:
    env = os.environ if env is None else env
    profiles: List[Dict[str, str]] = []
    default_url = (env.get("NAI_ENDPOINT") or "").strip()
    if default_url:
        profiles.append({
            "id": "default",
            "name": (env.get("NAI_PROFILE_DEFAULT_NAME") or "Padrão (manifest)").strip(),
            "endpoint_url": default_url,
            "model_name": (env.get("MODEL_NAME") or "").strip(),
            "api_key": (env.get("NAI_API_KEY") or "").strip(),
        })
    numbers = sorted({int(m.group(1)) for k in env for m in [_PROFILE_URL.match(k)] if m})
    for n in numbers:
        url = (env.get(f"NAI_PROFILE_{n}_URL") or "").strip()
        if not url:
            continue
        profiles.append({
            "id": str(n),
            "name": (env.get(f"NAI_PROFILE_{n}_NAME") or f"Endpoint {n}").strip(),
            "endpoint_url": url,
            "model_name": (env.get(f"NAI_PROFILE_{n}_MODEL") or "").strip(),
            "api_key": (env.get(f"NAI_PROFILE_{n}_API_KEY") or "").strip(),
        })
    return profiles


PROFILES = load_profiles()


def public_profiles() -> List[Dict[str, Any]]:
    return [
        {"id": p["id"], "name": p["name"], "endpoint_url": p["endpoint_url"], "model_name": p["model_name"],
         "has_api_key": bool(p["api_key"])}
        for p in PROFILES
    ]


def resolve_api_key(endpoint: str, api_key: Optional[str], profile_id: Optional[str] = None) -> str:
    """Chave digitada > chave do perfil escolhido > chave de outro perfil do mesmo host.

    Uma chave salva só é usada se o endpoint estiver no mesmo host do perfil, para
    nunca mandar a chave de um NAI para uma URL digitada que aponte para outro lugar.
    """
    if api_key and api_key.strip():
        return api_key.strip()
    host = endpoint_host(endpoint)
    if not host:
        return ""
    if profile_id:
        for p in PROFILES:
            if p["id"] == profile_id and p["api_key"] and endpoint_host(p["endpoint_url"]) == host:
                return p["api_key"]
    for p in PROFILES:
        if p["api_key"] and endpoint_host(p["endpoint_url"]) == host:
            return p["api_key"]
    return ""


def ssl_verify_setting():
    """Certificados autoassinados são comuns em ambientes NAI de demo.

    - NAI_CA_BUNDLE=/caminho/ca.pem  -> valida usando esse CA
    - NAI_VERIFY_SSL=true            -> valida com os CAs do sistema
    - padrão                         -> não valida (comportamento original)
    """
    ca_bundle = os.getenv("NAI_CA_BUNDLE")
    if ca_bundle:
        return ca_bundle
    return os.getenv("NAI_VERIFY_SSL", "false").strip().lower() in ("1", "true", "yes")


def make_client(timeout: Optional[httpx.Timeout] = None, limits: Optional[httpx.Limits] = None) -> httpx.AsyncClient:
    """Ponto único de criação do cliente HTTP (os testes substituem esta função)."""
    if timeout is None:
        timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
    kwargs = {"limits": limits} if limits is not None else {}
    return httpx.AsyncClient(verify=ssl_verify_setting(), timeout=timeout, **kwargs)


# ---------------------------------------------------------------------------
# Montagem da requisição
# ---------------------------------------------------------------------------

def models_url(chat_endpoint: str) -> str:
    """.../v1/chat/completions -> .../v1/models"""
    base = chat_endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    elif base.endswith("/completions"):
        base = base[: -len("/completions")]
    return base + "/models"


def auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def clean_history(history: List[Dict[str, Any]], limit: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    """Mantém só role/content de mensagens válidas e corta para as últimas N."""
    cleaned = []
    for msg in history or []:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    if limit > 0:
        cleaned = cleaned[-limit:]
    else:
        cleaned = []
    # A conversa deve começar com uma mensagem do usuário (alguns templates exigem)
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


def build_system_prompt(
    system_prompt: Optional[str],
    document_text: Optional[str],
    document_name: Optional[str],
) -> Tuple[str, bool]:
    """Retorna (prompt de sistema, documento_foi_truncado)."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    truncated = False
    if document_text and document_text.strip():
        text = document_text.strip()
        if DOC_MAX_CHARS > 0 and len(text) > DOC_MAX_CHARS:
            text = text[:DOC_MAX_CHARS]
            truncated = True
        name = document_name or "documento"
        prompt = (
            f"{prompt}\n\n"
            f"O usuário anexou o documento \"{name}\". Use o conteúdo abaixo como "
            f"referência principal para responder. Se a resposta não estiver no "
            f"documento, diga isso claramente.\n\n"
            f"<documento>\n{text}\n</documento>"
        )
    return prompt, truncated


def build_payload(
    user_input: str,
    history: List[Dict[str, Any]],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    document_text: Optional[str] = None,
    document_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    system, truncated = build_system_prompt(system_prompt, document_text, document_name)
    messages: List[Dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.extend(clean_history(history))
    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        "max_tokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
    }
    return payload, truncated


# ---------------------------------------------------------------------------
# Separação de raciocínio em tags <think>...</think>
# ---------------------------------------------------------------------------

class ThinkTagParser:
    """Separa raciocínio de resposta quando o modelo usa tags <think> no texto.

    Funciona em streaming (as tags podem chegar quebradas entre chunks).
    Emite tuplas (tipo, texto) com tipo em {"reasoning", "content", "reclassify"}.
    "reclassify" aparece quando chega um </think> sem <think> antes (caso do
    DeepSeek-R1, cujo template já abre a tag no prompt): tudo que foi emitido
    como resposta até ali era, na verdade, raciocínio.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.buffer = ""
        self.in_think = False
        self.seen_open = False
        self.seen_close = False
        self.emitted_content = False
        self.strip_leading = False  # remove quebras de linha logo após </think>

    def _content(self, out: List[Tuple[str, str]], text: str) -> None:
        if self.strip_leading:
            text = text.lstrip()
            if not text:
                return
            self.strip_leading = False
        out.append(("content", text))
        self.emitted_content = True

    def _hold_back(self, text: str, tag: str) -> int:
        """Quantos caracteres finais podem ser o início da tag (guardar para depois)."""
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if tag.startswith(text[-size:]):
                return size
        return 0

    def feed(self, text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        self.buffer += text
        while self.buffer:
            if self.in_think:
                idx = self.buffer.find(self.CLOSE)
                if idx == -1:
                    keep = self._hold_back(self.buffer, self.CLOSE)
                    emit = self.buffer[: len(self.buffer) - keep]
                    if emit:
                        out.append(("reasoning", emit))
                    self.buffer = self.buffer[len(emit):]
                    break
                if idx:
                    out.append(("reasoning", self.buffer[:idx]))
                self.buffer = self.buffer[idx + len(self.CLOSE):]
                self.in_think = False
                self.seen_close = True
                self.strip_leading = True
                continue

            # Fora do bloco de raciocínio
            if self.seen_close:
                # Depois do primeiro </think> tudo é resposta
                self._content(out, self.buffer)
                self.buffer = ""
                break

            open_idx = self.buffer.find(self.OPEN)
            close_idx = self.buffer.find(self.CLOSE)
            if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
                before = self.buffer[:open_idx]
                if before.strip():
                    self._content(out, before)
                self.buffer = self.buffer[open_idx + len(self.OPEN):]
                self.in_think = True
                self.seen_open = True
                continue
            if close_idx != -1:
                # </think> órfão: o que veio antes era raciocínio
                before = self.buffer[:close_idx]
                if self.emitted_content:
                    out.append(("reclassify", ""))
                if before:
                    out.append(("reasoning", before))
                self.buffer = self.buffer[close_idx + len(self.CLOSE):]
                self.seen_close = True
                self.strip_leading = True
                continue

            keep = max(self._hold_back(self.buffer, self.OPEN), self._hold_back(self.buffer, self.CLOSE))
            emit = self.buffer[: len(self.buffer) - keep]
            if emit:
                self._content(out, emit)
            self.buffer = self.buffer[len(emit):]
            break
        return out

    def flush(self) -> List[Tuple[str, str]]:
        if not self.buffer:
            return []
        text, self.buffer = self.buffer, ""
        if self.in_think:
            return [("reasoning", text)]
        out: List[Tuple[str, str]] = []
        self._content(out, text)
        return out


def split_reasoning(content: str) -> Tuple[str, str]:
    """Versão não-streaming: retorna (raciocínio, resposta)."""
    parser = ThinkTagParser()
    reasoning, answer = "", ""
    for kind, text in parser.feed(content) + parser.flush():
        if kind == "reasoning":
            reasoning += text
        elif kind == "content":
            answer += text
        elif kind == "reclassify":
            reasoning, answer = reasoning + answer, ""
    return reasoning.strip(), answer.strip()


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def build_metrics(
    start: float,
    first_token: Optional[float],
    end: float,
    usage: Optional[Dict[str, Any]],
    chunk_count: int,
) -> Dict[str, Any]:
    usage = usage or {}
    completion = usage.get("completion_tokens")
    prompt = usage.get("prompt_tokens")
    estimated = completion is None
    if estimated:
        # Sem "usage" no stream: cada chunk de vLLM costuma ser ~1 token
        completion = chunk_count
    total = usage.get("total_tokens")
    if total is None:
        total = (prompt or 0) + (completion or 0)

    ttft = (first_token - start) if first_token else None
    # Sem streaming (sem TTFT), tok/s é calculado sobre o tempo total
    gen_time = (end - first_token) if first_token else (end - start)
    tps = (completion / gen_time) if gen_time and gen_time > 0 and completion else 0.0

    return {
        "ttft": round(ttft, 3) if ttft is not None else None,
        "duration": round(end - start, 2),
        "tps": round(tps, 1),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "estimated": estimated,
    }


# ---------------------------------------------------------------------------
# Chamadas ao NAI
# ---------------------------------------------------------------------------

def _error_text(status: int, body: str) -> str:
    body = (body or "").strip()
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            err = data.get("error", data)
            if isinstance(err, dict):
                body = err.get("message") or err.get("detail") or json.dumps(err)
            else:
                body = str(err)
    except (ValueError, TypeError):
        pass
    hints = {
        401: " Verifique a API Key.",
        403: " A API Key não tem acesso a este endpoint.",
        404: " Verifique a URL do endpoint e o nome do modelo.",
    }
    return f"NAI respondeu HTTP {status}: {body[:500]}{hints.get(status, '')}"


def _delta_events(delta: Dict[str, Any], parser: ThinkTagParser) -> List[Tuple[str, str]]:
    events: List[Tuple[str, str]] = []
    # vLLM com reasoning parser: "reasoning_content" (antigo) ou "reasoning" (novo)
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        events.append(("reasoning", reasoning))
    content = delta.get("content")
    if isinstance(content, str) and content:
        events.extend(parser.feed(content))
    return events


async def stream_chat(endpoint: str, api_key: Optional[str], payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    """Gera eventos: reasoning, content, reclassify, metrics, error, done."""
    start = time.perf_counter()
    first_token: Optional[float] = None
    chunk_count = 0
    usage: Optional[Dict[str, Any]] = None
    parser = ThinkTagParser()

    def mark_first():
        nonlocal first_token
        if first_token is None:
            first_token = time.perf_counter()

    attempts = [
        {**payload, "stream": True, "stream_options": {"include_usage": True}},
        {**payload, "stream": True},  # fallback se o endpoint rejeitar stream_options
    ]

    try:
        async with make_client() as client:
            for i, body in enumerate(attempts):
                async with client.stream("POST", endpoint, json=body, headers=auth_headers(api_key)) as resp:
                    if resp.status_code != 200:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        if i == 0 and resp.status_code in (400, 422):
                            continue
                        yield {"type": "error", "message": _error_text(resp.status_code, text)}
                        return

                    ctype = resp.headers.get("content-type", "")
                    if "text/event-stream" not in ctype:
                        # Endpoint ignorou o stream e devolveu JSON completo
                        data = json.loads(await resp.aread())
                        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
                        for kind, text in _delta_events(msg, parser) + parser.flush():
                            yield {"type": kind, "content": text}
                        end = time.perf_counter()
                        yield {"type": "metrics", "metrics": build_metrics(start, None, end, data.get("usage"), 0)}
                        yield {"type": "done"}
                        return

                    async for line in resp.aiter_lines():
                        line = line.strip()
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
                            yield {"type": "error", "message": f"Erro do NAI durante a geração: {msg}"}
                            return
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            events = _delta_events(delta, parser)
                            if events:
                                mark_first()
                                chunk_count += 1
                            for kind, text in events:
                                yield {"type": kind, "content": text}

                    for kind, text in parser.flush():
                        yield {"type": kind, "content": text}
                    end = time.perf_counter()
                    yield {"type": "metrics", "metrics": build_metrics(start, first_token, end, usage, chunk_count)}
                    yield {"type": "done"}
                    return
    except httpx.ConnectError as e:
        yield {"type": "error", "message": f"Não foi possível conectar ao endpoint NAI: {e}"}
    except httpx.TimeoutException:
        yield {"type": "error", "message": "Tempo esgotado esperando o NAI responder."}
    except Exception as e:  # noqa: BLE001 - o erro precisa chegar ao usuário
        yield {"type": "error", "message": f"Erro inesperado: {type(e).__name__}: {e}"}


async def complete_chat(endpoint: str, api_key: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Versão sem streaming (mantida para o endpoint /ask)."""
    start = time.perf_counter()
    try:
        async with make_client() as client:
            resp = await client.post(endpoint, json={**payload, "stream": False}, headers=auth_headers(api_key))
        end = time.perf_counter()
        if resp.status_code != 200:
            return {"status": "error", "response": _error_text(resp.status_code, resp.text)}
        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        reasoning_field = msg.get("reasoning_content") or msg.get("reasoning") or ""
        tag_reasoning, answer = split_reasoning(msg.get("content") or "")
        reasoning = "\n".join(r for r in (reasoning_field, tag_reasoning) if r).strip()
        metrics = build_metrics(start, None, end, data.get("usage"), 0)
        return {"status": "success", "response": answer, "reasoning": reasoning, "metrics": metrics}
    except httpx.ConnectError as e:
        return {"status": "error", "response": f"Não foi possível conectar ao endpoint NAI: {e}"}
    except httpx.TimeoutException:
        return {"status": "error", "response": "Tempo esgotado esperando o NAI responder."}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "response": f"Erro inesperado: {type(e).__name__}: {e}"}


async def list_models(endpoint: str, api_key: Optional[str]) -> Dict[str, Any]:
    url = models_url(endpoint)
    try:
        async with make_client(httpx.Timeout(20.0, connect=CONNECT_TIMEOUT)) as client:
            resp = await client.get(url, headers=auth_headers(api_key))
        if resp.status_code != 200:
            return {"status": "error", "models": [], "message": _error_text(resp.status_code, resp.text), "url": url}
        data = resp.json()
        items = data.get("data", data) if isinstance(data, dict) else data
        models = sorted(
            {m.get("id") for m in items if isinstance(m, dict) and m.get("id")}
            if isinstance(items, list) else set()
        )
        if not models:
            return {"status": "error", "models": [], "message": "O endpoint não retornou nenhum modelo.", "url": url}
        return {"status": "success", "models": models, "url": url}
    except httpx.ConnectError as e:
        return {"status": "error", "models": [], "message": f"Falha de conexão: {e}", "url": url}
    except httpx.TimeoutException:
        return {"status": "error", "models": [], "message": "Tempo esgotado.", "url": url}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "models": [], "message": f"{type(e).__name__}: {e}", "url": url}


async def check_connectivity(endpoint: str, api_key: Optional[str]) -> Dict[str, Any]:
    """Testa rede, TLS e autenticação com uma chamada HTTP leve (GET /models)."""
    url = models_url(endpoint)
    start = time.perf_counter()
    try:
        async with make_client(httpx.Timeout(15.0, connect=CONNECT_TIMEOUT)) as client:
            resp = await client.get(url, headers=auth_headers(api_key))
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        code = resp.status_code
        if code == 200:
            return {"reachable": True, "ok": True, "status_code": code, "latency_ms": latency_ms,
                    "message": f"Endpoint acessível e autenticado ({latency_ms} ms)."}
        if code in (401, 403):
            msg = f"Endpoint acessível ({latency_ms} ms), mas a autenticação falhou (HTTP {code}). Verifique a API Key."
        elif code == 404:
            msg = (f"Servidor acessível ({latency_ms} ms), mas a rota de modelos não existe (HTTP 404). "
                   f"O chat ainda pode funcionar; confira a URL do endpoint.")
        else:
            msg = f"Servidor acessível ({latency_ms} ms), mas respondeu HTTP {code}."
        return {"reachable": True, "ok": False, "status_code": code, "latency_ms": latency_ms, "message": msg}
    except httpx.ConnectError as e:
        text = str(e)
        if "CERTIFICATE" in text.upper() or "SSL" in text.upper():
            msg = f"Falha de TLS ao conectar: {text}. Configure NAI_CA_BUNDLE ou NAI_VERIFY_SSL=false."
        else:
            msg = f"Não foi possível conectar ao host: {text}"
        return {"reachable": False, "ok": False, "status_code": None, "latency_ms": None, "message": msg}
    except httpx.TimeoutException:
        return {"reachable": False, "ok": False, "status_code": None, "latency_ms": None,
                "message": "Tempo esgotado: o host não respondeu (rede, firewall ou DNS)."}
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "ok": False, "status_code": None, "latency_ms": None,
                "message": f"{type(e).__name__}: {e}"}
