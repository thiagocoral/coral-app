"""NTNX BR AI Assistant - interface de chat para endpoints do Nutanix Enterprise AI.

Fala com o backend (core/) em http://app-service, que por sua vez chama o NAI.
Recursos: streaming real, métricas por resposta (TTFT, tok/s, tokens), raciocínio
recolhível, parâmetros do modelo, lista de modelos, teste de conexão, modo de
comparação entre dois endpoints, documento anexado e exportação da conversa.
"""

import io
import json
import os
import queue
import threading
import time
from datetime import datetime
from urllib.parse import urlsplit

import requests
import streamlit as st

import bench_charts as bc

BACKEND_URL = os.getenv("BACKEND_URL", "http://app-service").rstrip("/")
API = f"{BACKEND_URL}/api/v1/chat"
UI_REFRESH_SECONDS = 0.05
MODE_CHAT = "💬 Chat"
MODE_COMPARE = "⚖️ Comparação"
MODE_BENCH = "📈 Benchmark"
ORDER_PARALLEL = "Simultâneo"
ORDER_SEQUENTIAL = "Sequencial (A, depois B)"
BENCH_SEQUENTIAL = "Um endpoint por vez (mais preciso)"
BENCH_PARALLEL = "Os dois ao mesmo tempo"
KIND_SINGLE = "Rodada única"
KIND_CONC = "Varredura de concorrência"
KIND_CTX = "Varredura de contexto"
DOC_TYPES = ["pdf", "txt", "md", "csv", "json", "log", "yaml", "yml"]

FALLBACK_DEFAULTS = {
    "endpoint_url": "https://10.42.163.114/enterpriseai/gateway/v1/chat/completions",
    "model_name": "unified-endpoint",
    "system_prompt": "Você é o assistente NTNX BR, rodando no Nutanix Enterprise AI (NAI). "
                     "Responda de forma clara e objetiva, em português do Brasil.",
    "temperature": 0.3,
    "max_tokens": 2048,
    "doc_max_chars": 20000,
    "benchmark_prompt": "Explique, em cerca de 300 palavras, o que é Kubernetes e por que ele é usado "
                        "para rodar aplicações de IA em produção.",
}

st.set_page_config(
    page_title="NTNX BR AI - Powered by Nutanix NAI and NKP",
    page_icon="🤖",
    layout="wide" if st.session_state.get("mode") in (MODE_COMPARE, MODE_BENCH) else "centered",
)

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Comunicação com o backend
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def backend_defaults(api):
    """Padrões configurados no backend. Retorna (dados, online)."""
    try:
        resp = requests.get(f"{api}/defaults", timeout=3)
        if resp.status_code == 200:
            return {**FALLBACK_DEFAULTS, **resp.json()}, True
    except requests.RequestException:
        pass
    return dict(FALLBACK_DEFAULTS), False


@st.cache_data(ttl=60, show_spinner=False)
def backend_profiles(api):
    """Endpoints salvos no servidor (sem as API Keys)."""
    try:
        resp = requests.get(f"{api}/profiles", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("profiles", [])
    except requests.RequestException:
        pass
    return []


def endpoint_host(url):
    """scheme://host:porta, igual ao backend: uma chave só vale para o mesmo host."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


# Chaves vindas do ambiente do próprio frontend: ficam no servidor do Streamlit e
# não são mais preenchidas no campo da tela (não chegam ao navegador).
FRONTEND_KEYS = {}
for _url_var, _key_var in (("NAI_ENDPOINT_B", "NAI_API_KEY_B"), ("NAI_ENDPOINT", "NAI_API_KEY")):
    _host = endpoint_host(os.getenv(_url_var))
    if _host and os.getenv(_key_var):
        FRONTEND_KEYS[_host] = os.getenv(_key_var)


def post_json(path, body, timeout=30):
    try:
        resp = requests.post(f"{API}/{path}", json=body, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Backend respondeu HTTP {resp.status_code}: {resp.text[:300]}"}
    except requests.RequestException as e:
        return {"error": f"Não foi possível falar com o backend ({BACKEND_URL}): {e}"}


def stream_events(payload, path="stream", read_timeout=600):
    """Lê o NDJSON do backend e entrega um evento (dict) por vez."""
    try:
        with requests.post(f"{API}/{path}", json=payload, stream=True, timeout=(10, read_timeout)) as resp:
            if resp.status_code != 200:
                yield {"type": "error", "message": f"Backend respondeu HTTP {resp.status_code}: {resp.text[:300]}"}
                return
            buffer = b""
            for chunk in resp.iter_content(chunk_size=None):
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        yield json.loads(line.decode("utf-8"))
            if buffer.strip():
                yield json.loads(buffer.decode("utf-8"))
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "message": f"Não foi possível conectar ao backend em {BACKEND_URL}."}
    except requests.exceptions.Timeout:
        yield {"type": "error", "message": "Tempo esgotado esperando o backend."}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"Erro lendo a resposta: {type(e).__name__}: {e}"}


class Answer:
    """Acumula os eventos do stream de uma resposta."""

    def __init__(self):
        self.reasoning = ""
        self.content = ""
        self.metrics = None
        self.error = None
        self.notice = None
        self.done = False

    def apply(self, event):
        kind = event.get("type")
        if kind == "reasoning":
            self.reasoning += event.get("content", "")
        elif kind == "content":
            self.content += event.get("content", "")
        elif kind == "reclassify":
            self.reasoning, self.content = self.reasoning + self.content, ""
        elif kind == "metrics":
            self.metrics = event.get("metrics")
        elif kind == "notice":
            self.notice = event.get("message")
        elif kind == "error":
            self.error = event.get("message")
            self.done = True
        elif kind == "done":
            self.done = True

    def as_message(self):
        return {
            "role": "assistant",
            "content": self.content.strip(),
            "reasoning": self.reasoning.strip(),
            "metrics": self.metrics,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------

def init_state():
    d, _ = backend_defaults(API)
    endpoint = os.getenv("NAI_ENDPOINT") or d["endpoint_url"]
    model = os.getenv("MODEL_NAME") or d["model_name"]
    defaults = {
        "mode": MODE_CHAT,
        "endpoint_A": endpoint,
        "key_A": "",
        "model_A": model,
        "endpoint_B": os.getenv("NAI_ENDPOINT_B") or endpoint,
        "key_B": "",
        "model_B": os.getenv("MODEL_NAME_B") or model,
        "system_prompt": d["system_prompt"],
        "temperature": float(d["temperature"]),
        "max_tokens": int(d["max_tokens"]),
        "messages": [],
        "cmp_A": [],
        "cmp_B": [],
        "models_A": [],
        "models_B": [],
        "conn_A": None,
        "conn_B": None,
        "total_tokens": 0,
        "last_metrics": {},
        "doc": None,
        "compare_order": ORDER_PARALLEL,
        "bench_include_b": False,
        "bench_prompt": d.get("benchmark_prompt") or FALLBACK_DEFAULTS["benchmark_prompt"],
        "bench_n": 10,
        "bench_conc": 1,
        "bench_warmup": 1,
        "bench_max_tokens": 256,
        "bench_temp": 0.0,
        "bench_avoid_cache": True,
        "bench_ignore_eos": False,
        "bench_order": BENCH_SEQUENTIAL,
        "bench_run": None,
        "profile_A": None,
        "profile_B": None,
        "saved_A": "",
        "saved_B": "",
        "recent_endpoints": [],
        "metrics_A": "",
        "metrics_B": "",
        "mcheck_A": None,
        "mcheck_B": None,
        "bench_kind": KIND_SINGLE,
        "bench_levels_conc": "1, 2, 4, 8, 16",
        "bench_per_level": 4,
        "bench_levels_ctx": "256, 1024, 4096, 8192",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
ss = st.session_state

# O Streamlit descarta o valor de um widget quando ele não é desenhado na
# execução. Reatribuir mantém configurações ao trocar de modo (ex.: endpoint B,
# system prompt). Não vale para file_uploader, que não aceita atribuição.
PERSISTENT_KEYS = (
    "mode", "endpoint_A", "key_A", "model_A", "endpoint_B", "key_B", "model_B", "pick_A", "pick_B",
    "system_prompt", "temperature", "max_tokens", "compare_order", "bench_include_b", "bench_prompt",
    "bench_n", "bench_conc", "bench_warmup", "bench_max_tokens", "bench_temp", "bench_avoid_cache",
    "bench_ignore_eos", "bench_order", "metrics_A", "metrics_B", "bench_kind", "bench_levels_conc",
    "bench_per_level", "bench_levels_ctx",
)
for _key in PERSISTENT_KEYS:
    if _key in ss:
        ss[_key] = ss[_key]


def key_source(side):
    """De onde vem a API Key deste endpoint: (chave a enviar, descrição ou None)."""
    typed = (ss[f"key_{side}"] or "").strip()
    if typed:
        return typed, None
    host = endpoint_host(ss[f"endpoint_{side}"])
    if side == "B" and ss["key_A"].strip() and host and host == endpoint_host(ss["endpoint_A"]):
        return ss["key_A"].strip(), "Usando a mesma API Key do endpoint A (mesmo host)."
    if host and host in FRONTEND_KEYS:
        return FRONTEND_KEYS[host], "🔑 Usando a API Key configurada no servidor para este host."
    if host and any(p.get("has_api_key") and endpoint_host(p["endpoint_url"]) == host
                    for p in backend_profiles(API)):
        return "", "🔑 Usando a API Key salva no servidor para este host."
    return "", None


def endpoint_config(side):
    return {
        "endpoint_url": ss[f"endpoint_{side}"],
        "api_key": key_source(side)[0],
        "model_name": ss[f"model_{side}"],
        "profile": ss[f"profile_{side}"],
    }


def remember_endpoint(side):
    """Guarda o endpoint usado com sucesso na lista de recentes desta sessão."""
    entry = {
        "url": ss[f"endpoint_{side}"].strip(),
        "model": ss[f"model_{side}"].strip(),
        "key": ss[f"key_{side}"].strip(),
        "profile": ss[f"profile_{side}"],
        "metrics": ss[f"metrics_{side}"].strip(),
    }
    if not entry["url"]:
        return
    recents = [r for r in ss["recent_endpoints"] if (r["url"], r["model"]) != (entry["url"], entry["model"])]
    ss["recent_endpoints"] = ([entry] + recents)[:8]


def build_history(messages):
    """Pares usuário/assistente que deram certo (ignora respostas com erro)."""
    history, pending_user = [], None
    for msg in messages:
        if msg["role"] == "user":
            pending_user = msg
        elif pending_user and not msg.get("error") and msg.get("content"):
            history.append({"role": "user", "content": pending_user["content"]})
            history.append({"role": "assistant", "content": msg["content"]})
            pending_user = None
    return history


def build_payload(prompt, side, messages):
    payload = {
        "user_input": prompt,
        "history": build_history(messages),
        "system_prompt": ss["system_prompt"],
        "temperature": float(ss["temperature"]),
        "max_tokens": int(ss["max_tokens"]),
        **endpoint_config(side),
    }
    if ss["doc"]:
        payload["document_text"] = ss["doc"]["text"]
        payload["document_name"] = ss["doc"]["name"]
    return payload


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------

def fmt_metrics(m):
    if not m:
        return ""
    approx = "~" if m.get("estimated") else ""
    parts = []
    if m.get("ttft") is not None:
        parts.append(f"⏱️ 1º token {m['ttft']:.2f}s")
    parts.append(f"⚡ {approx}{m.get('tps', 0)} tok/s")
    prompt_t = m.get("prompt_tokens")
    comp_t = m.get("completion_tokens")
    if prompt_t is not None:
        parts.append(f"🧮 {prompt_t} → {approx}{comp_t} tokens")
    elif comp_t is not None:
        parts.append(f"🧮 {approx}{comp_t} tokens gerados")
    parts.append(f"🕒 {m.get('duration', 0)}s total")
    return " · ".join(parts)


def render_assistant(msg):
    if msg.get("reasoning"):
        with st.expander("💭 Raciocínio do modelo", expanded=False):
            st.markdown(msg["reasoning"])
    if msg.get("error"):
        if msg.get("content"):
            st.markdown(msg["content"])
        st.error(msg["error"])
    else:
        st.markdown(msg.get("content") or "_(resposta vazia)_")
    if msg.get("metrics"):
        st.caption(fmt_metrics(msg["metrics"]))


def render_messages(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg)


def conversation_markdown(sections):
    """sections: lista de (título, config, mensagens)."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = ["# Conversa — NTNX BR AI Assistant", "", f"_Exportado em {now}_", ""]
    if ss["doc"]:
        lines += [f"**Documento anexado:** {ss['doc']['name']}", ""]
    lines += ["**System prompt:**", "", f"> {ss['system_prompt']}", "",
              f"**Temperatura:** {ss['temperature']} · **Máx. tokens:** {ss['max_tokens']}", ""]
    for title, cfg, messages in sections:
        if title:
            lines += [f"## {title}", ""]
        lines += [f"**Endpoint:** `{cfg['endpoint_url']}`  ", f"**Modelo:** `{cfg['model_name']}`", ""]
        for msg in messages:
            if msg["role"] == "user":
                lines += ["### 🧑 Usuário", "", msg["content"], ""]
            else:
                lines += ["### 🤖 Assistente", ""]
                if msg.get("reasoning"):
                    lines += ["<details><summary>Raciocínio</summary>", "", msg["reasoning"], "", "</details>", ""]
                if msg.get("content"):
                    lines += [msg["content"], ""]
                if msg.get("error"):
                    lines += [f"> ⚠️ Erro: {msg['error']}", ""]
                if msg.get("metrics"):
                    lines += [f"_{fmt_metrics(msg['metrics'])}_", ""]
    return "\n".join(lines)


def extract_document(uploaded):
    data = uploaded.getvalue()
    name = uploaded.name
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(p.strip() for p in pages if p.strip())
        return {"name": name, "text": text, "pages": len(reader.pages)}
    for encoding in ("utf-8", "latin-1"):
        try:
            return {"name": name, "text": data.decode(encoding), "pages": None}
        except UnicodeDecodeError:
            continue
    return {"name": name, "text": "", "pages": None}


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

def pick_model(side):
    ss[f"model_{side}"] = ss[f"pick_{side}"]


def typed_model(side):
    """Mantém o seletor coerente quando o nome do modelo é digitado à mão."""
    if ss[f"models_{side}"]:
        value = ss[f"model_{side}"]
        ss[f"pick_{side}"] = value if value in ss[f"models_{side}"] else None


def saved_options():
    options = [""]
    labels = {"": "Selecionar..."}
    for p in backend_profiles(API):
        value = f"p:{p['id']}"
        options.append(value)
        model = f" · {p['model_name']}" if p.get("model_name") else ""
        labels[value] = f"⭐ {p['name']}{model}"
    for i, r in enumerate(ss["recent_endpoints"]):
        value = f"r:{i}"
        options.append(value)
        host = endpoint_host(r["url"]).split("://")[-1] or r["url"]
        labels[value] = f"🕘 {r['model'] or 'modelo padrão'} @ {host}"
    return options, labels


def apply_saved(side):
    choice = ss[f"saved_{side}"]
    entry = None
    if choice.startswith("p:"):
        p = next((p for p in backend_profiles(API) if p["id"] == choice[2:]), None)
        if p:
            entry = {"url": p["endpoint_url"], "model": p["model_name"], "key": "", "profile": p["id"],
                     "metrics": p.get("metrics_url") or ""}
    elif choice.startswith("r:"):
        idx = int(choice[2:])
        if idx < len(ss["recent_endpoints"]):
            entry = dict(ss["recent_endpoints"][idx])
    if entry:
        ss[f"endpoint_{side}"] = entry["url"]
        if entry["model"]:
            ss[f"model_{side}"] = entry["model"]
        ss[f"key_{side}"] = entry["key"]
        ss[f"profile_{side}"] = entry["profile"]
        if entry.get("metrics"):
            ss[f"metrics_{side}"] = entry["metrics"]
        ss[f"mcheck_{side}"] = None
        ss[f"models_{side}"] = []
        ss.pop(f"pick_{side}", None)
        ss[f"conn_{side}"] = None
    ss[f"saved_{side}"] = ""  # o seletor funciona como menu: volta ao início


def endpoint_controls(side):
    options, labels = saved_options()
    if len(options) > 1:
        st.selectbox("Endpoints salvos", options, key=f"saved_{side}", format_func=labels.get,
                     on_change=apply_saved, args=(side,),
                     help="⭐ salvos no servidor (variáveis de ambiente) · 🕘 usados nesta sessão")
    st.text_input("URL do endpoint", key=f"endpoint_{side}")
    st.text_input("API Key", key=f"key_{side}", type="password",
                  help="Em branco: usa a chave salva no servidor para este host, se houver.")
    note = key_source(side)[1]
    if note:
        st.caption(note)

    models = ss[f"models_{side}"]
    if models:
        current = ss[f"model_{side}"]
        if f"pick_{side}" not in ss:
            ss[f"pick_{side}"] = current if current in models else None
        st.selectbox(
            "Modelos disponíveis",
            models,
            placeholder="Escolha um modelo...",
            key=f"pick_{side}",
            on_change=pick_model,
            args=(side,),
        )
    st.text_input("Modelo", key=f"model_{side}", on_change=typed_model, args=(side,))

    col1, col2 = st.columns(2)
    if col1.button("🔄 Listar", key=f"load_{side}", help="Buscar os modelos do endpoint", width="stretch"):
        with st.spinner("Buscando modelos..."):
            cfg = endpoint_config(side)
            result = post_json("models", {"endpoint_url": cfg["endpoint_url"], "api_key": cfg["api_key"],
                                          "profile": cfg["profile"]})
        if result.get("status") == "success":
            remember_endpoint(side)
            ss[f"models_{side}"] = result["models"]
            current = ss[f"model_{side}"]
            ss[f"pick_{side}"] = current if current in result["models"] else None
            ss[f"conn_{side}"] = {"kind": "success", "text": f"{len(result['models'])} modelo(s) encontrado(s)."}
        else:
            ss[f"conn_{side}"] = {"kind": "warning",
                                  "text": result.get("message") or result.get("error") or "Falha ao listar modelos."}
        st.rerun()
    if col2.button("🩺 Testar", key=f"test_{side}", help="Testar rede, TLS e API Key", width="stretch"):
        with st.spinner("Testando conexão..."):
            cfg = endpoint_config(side)
            result = post_json("connectivity", {"endpoint_url": cfg["endpoint_url"], "api_key": cfg["api_key"],
                                                "profile": cfg["profile"]})
        if "error" in result:
            ss[f"conn_{side}"] = {"kind": "error", "text": result["error"]}
        else:
            if result.get("ok"):
                remember_endpoint(side)
            kind = "success" if result.get("ok") else ("warning" if result.get("reachable") else "error")
            ss[f"conn_{side}"] = {"kind": kind, "text": result.get("message", "")}

    status = ss[f"conn_{side}"]
    if status:
        getattr(st, status["kind"])(status["text"])

    if ss["mode"] == MODE_BENCH:
        st.text_input(
            "URL de métricas do servidor (opcional)", key=f"metrics_{side}",
            placeholder="http://<pod-do-vllm>:8000/metrics",
            help="Endpoint /metrics do servidor de inferência (vLLM). Com ele o benchmark mostra uso do KV "
                 "cache, fila e cache de prefixo. Precisa ser acessível a partir do backend do app.",
        )
        if ss[f"metrics_{side}"].strip() and st.button("📈 Testar métricas", key=f"mtest_{side}", width="stretch"):
            with st.spinner("Lendo métricas..."):
                ss[f"mcheck_{side}"] = post_json("metrics-check", {"metrics_url": ss[f"metrics_{side}"].strip(),
                                                                   "model_name": ss[f"model_{side}"]})
        check = ss[f"mcheck_{side}"]
        if check:
            if check.get("ok"):
                kv = check["metrics"].get("Uso do KV cache")
                st.success("Métricas encontradas: " + ", ".join(check["metrics"])
                           + (f". KV cache agora: {kv:.1f}%" if kv is not None else "."))
            else:
                st.warning(check.get("error") or "Não foi possível ler as métricas.")


def current_namespace():
    """Namespace do próprio pod, quando roda no Kubernetes."""
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", encoding="utf-8") as f:
            return f.read().strip() or "<namespace>"
    except OSError:
        return "<namespace>"


def shell_quote(value):
    return "'" + value.replace("'", "'\\''") + "'"


def profiles_snippet(entries, start, include_keys):
    """Trecho de env do deployment.yaml e comando do Secret para salvar endpoints."""
    env_lines, secret_args = [], []
    for offset, e in enumerate(entries):
        n = start + offset
        host = endpoint_host(e["url"]).split("://")[-1] or e["url"]
        name = f"{e['model']} @ {host}" if e["model"] else host
        env_lines += [
            f"- name: NAI_PROFILE_{n}_NAME",
            f"  value: {json.dumps(name, ensure_ascii=False)}",
            f"- name: NAI_PROFILE_{n}_URL",
            f"  value: {json.dumps(e['url'], ensure_ascii=False)}",
        ]
        if e["model"]:
            env_lines += [f"- name: NAI_PROFILE_{n}_MODEL", f"  value: {json.dumps(e['model'], ensure_ascii=False)}"]
        if e.get("metrics"):
            env_lines += [f"- name: NAI_PROFILE_{n}_METRICS_URL",
                          f"  value: {json.dumps(e['metrics'], ensure_ascii=False)}"]
        if e["key"]:
            env_lines += [
                f"- name: NAI_PROFILE_{n}_API_KEY",
                "  valueFrom:",
                "    secretKeyRef:",
                "      name: nai-endpoints",
                f"      key: profile-{n}-api-key",
                "      optional: true",
            ]
            value = e["key"] if include_keys else "<API_KEY>"
            secret_args.append(f"  --from-literal=profile-{n}-api-key={shell_quote(value)} \\")
    secret_cmd = None
    if secret_args:
        secret_cmd = "\n".join(
            [f"kubectl create secret generic nai-endpoints -n {current_namespace()} \\"] + secret_args
            + ["  --dry-run=client -o yaml | kubectl apply -f -"]
        )
    return "\n".join(env_lines), secret_cmd


def render_profile_export():
    """Mostra como tornar permanentes os endpoints usados nesta sessão."""
    saved = {(p["endpoint_url"], p.get("model_name") or "") for p in backend_profiles(API)}
    entries = [r for r in ss["recent_endpoints"] if (r["url"], r["model"]) not in saved]
    with st.expander("🗂️ Salvar endpoints no Kubernetes"):
        st.caption("Endpoints salvos ficam em variáveis de ambiente do backend e aparecem para todos com ⭐. "
                   "As API Keys ficam num Secret e nunca são enviadas ao navegador.")
        if not entries:
            st.caption("Os endpoints que você usar com sucesso nesta sessão aparecem aqui, prontos para salvar.")
            return
        ids = [int(p["id"]) for p in backend_profiles(API) if str(p["id"]).isdigit()]
        include_keys = st.checkbox("Mostrar as API Keys no comando", key="export_keys",
                                   help="Desmarcado, o comando usa <API_KEY> no lugar das chaves.")
        env_yaml, secret_cmd = profiles_snippet(entries, max(ids, default=0) + 1, include_keys)
        st.markdown("**1. No `deployment.yaml`, em `env:` do container `app-container`:**")
        st.code(env_yaml, language="yaml")
        if secret_cmd:
            st.markdown("**2. Crie ou atualize o Secret com as chaves:**")
            st.code(secret_cmd, language="bash")
        st.caption("Depois do deploy (ou de reiniciar o backend), os endpoints aparecem como ⭐ no seletor.")


with st.sidebar:
    st.title("⚙️ Configurações de IA")
    defaults, backend_online = backend_defaults(API)
    if not backend_online:
        st.warning(f"Backend indisponível em {BACKEND_URL}.")

    st.radio("Modo", [MODE_CHAT, MODE_COMPARE, MODE_BENCH], key="mode", horizontal=True)
    comparing = ss["mode"] == MODE_COMPARE
    benching = ss["mode"] == MODE_BENCH
    if benching:
        st.checkbox("Incluir endpoint B", key="bench_include_b",
                    help="Mede dois endpoints ou modelos com as mesmas configurações.")
    show_b = comparing or (benching and ss["bench_include_b"])

    with st.expander("🔌 Endpoint A" if show_b else "🔌 Endpoint NAI", expanded=True):
        st.caption("Valores iniciais carregados do Kubernetes.")
        endpoint_controls("A")
    if show_b:
        with st.expander("🔌 Endpoint B", expanded=True):
            endpoint_controls("B")
    if comparing:
        st.radio(
            "Envio das perguntas", [ORDER_PARALLEL, ORDER_SEQUENTIAL], key="compare_order",
            help="Simultâneo: os dois recebem a pergunta ao mesmo tempo. Sequencial: B só começa quando A "
                 "termina, para que um não dispute recursos com o outro (útil quando estão no mesmo cluster/GPU).",
        )
    render_profile_export()

if benching:
    # O seletor de arquivo não aparece neste modo; ao voltar ele vem vazio,
    # mas o documento anexado deve continuar ativo.
    ss["uploader_active"] = False
else:
    with st.sidebar:
        with st.expander("🎛️ Parâmetros do modelo"):
            st.text_area("System prompt", key="system_prompt", height=140)
            st.slider("Temperatura", 0.0, 2.0, step=0.05, key="temperature",
                      help="Mais baixo: respostas mais previsíveis. Mais alto: mais criativas.")
            st.number_input("Máximo de tokens na resposta", min_value=16, max_value=131072, step=256, key="max_tokens")

        with st.expander("📎 Documento", expanded=bool(ss["doc"])):
            uploaded = st.file_uploader("Pergunte sobre um arquivo", type=DOC_TYPES, key="uploader")
            if uploaded is None:
                # Só remove se o usuário tirou o arquivo; ao voltar de outro modo o
                # seletor aparece vazio, mas o documento continua ativo.
                if ss.get("uploader_active"):
                    ss["doc"] = None
            elif not ss["doc"] or ss["doc"]["name"] != uploaded.name or ss["doc"].get("size") != uploaded.size:
                try:
                    doc = extract_document(uploaded)
                    doc["size"] = uploaded.size
                    ss["doc"] = doc
                except Exception as e:  # noqa: BLE001
                    ss["doc"] = None
                    st.error(f"Não foi possível ler o arquivo: {e}")
            ss["uploader_active"] = uploaded is not None
            if ss["doc"] and uploaded is None:
                if st.button("✖️ Remover documento", key="remove_doc"):
                    ss["doc"] = None
                    st.rerun()
            if ss["doc"]:
                doc = ss["doc"]
                chars = len(doc["text"])
                pages = f" · {doc['pages']} página(s)" if doc.get("pages") else ""
                if chars == 0:
                    st.warning("Nenhum texto extraído. PDFs escaneados (imagem) não são suportados.")
                else:
                    st.success(f"{doc['name']}: {chars:,} caracteres{pages}. O conteúdo vai junto em cada pergunta.")
                    limit = int(defaults.get("doc_max_chars") or 0)
                    if limit and chars > limit:
                        st.warning(f"Só os primeiros {limit:,} caracteres serão enviados ao modelo.")

        with st.expander("💾 Conversa", expanded=True):
            if comparing:
                sections = [("Endpoint A", endpoint_config("A"), ss["cmp_A"]),
                            ("Endpoint B", endpoint_config("B"), ss["cmp_B"])]
                has_messages = bool(ss["cmp_A"])
            else:
                sections = [("", endpoint_config("A"), ss["messages"])]
                has_messages = bool(ss["messages"])
            col1, col2 = st.columns(2)
            if col1.button("🗑️ Limpar", width="stretch", disabled=not has_messages):
                if comparing:
                    ss["cmp_A"], ss["cmp_B"] = [], []
                else:
                    ss["messages"] = []
                ss["total_tokens"] = 0
                ss["last_metrics"] = {}
                st.rerun()
            col2.download_button(
                "⬇️ Exportar",
                data=conversation_markdown(sections),
                file_name=f"conversa-ntnx-ai-{datetime.now().strftime('%Y%m%d-%H%M')}.md",
                mime="text/markdown",
                width="stretch",
                disabled=not has_messages,
            )

        st.subheader("📊 Métricas")
        st.metric("Tokens acumulados", f"{ss['total_tokens']:,}", help="Zera ao limpar a conversa.")
        for label, m in ss["last_metrics"].items():
            st.caption(f"**{label}** {fmt_metrics(m)}")


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------

st.title("🤖 NTNX BR AI Assistant")
st.caption("🚀 Rodando no NKP via **Nutanix Enterprise AI (NAI)**")
if ss.get("pending_toast"):
    st.toast(ss.pop("pending_toast"), icon="📎")
if ss["doc"] and ss["doc"]["text"] and ss["mode"] != MODE_BENCH:
    st.info(f"📎 Respondendo com base em **{ss['doc']['name']}**")


def live_render(answer, slots, final=False):
    """Atualiza os placeholders de uma resposta em andamento."""
    if answer.reasoning:
        if slots.get("reasoning") is None:
            slots["reasoning"] = slots["reasoning_box"].expander("💭 Raciocínio do modelo", expanded=True).empty()
        slots["reasoning"].markdown(answer.reasoning)
    if answer.content:
        slots["answer"].markdown(answer.content + ("" if final else " ▌"))
    elif answer.reasoning and not final:
        slots["answer"].markdown("_Raciocinando..._")
    if answer.metrics:
        slots["metrics"].caption(fmt_metrics(answer.metrics))


def assistant_slots():
    slots = {"reasoning_box": st.container(), "reasoning": None}
    slots["answer"] = st.empty()
    slots["metrics"] = st.empty()
    slots["answer"].markdown("_Pensando..._")
    return slots


def finish(answer, label, store, side):
    if answer.notice:
        ss["pending_toast"] = answer.notice
    msg = answer.as_message()
    store.append(msg)
    if not answer.error and answer.content:
        remember_endpoint(side)
    if answer.metrics:
        ss["total_tokens"] += int(answer.metrics.get("total_tokens") or 0)
        ss["last_metrics"][label] = answer.metrics


def run_chat(prompt):
    history = list(ss["messages"])
    ss["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        slots = assistant_slots()
        answer = Answer()
        last = 0.0
        for event in stream_events(build_payload(prompt, "A", history)):
            answer.apply(event)
            now = time.monotonic()
            if now - last >= UI_REFRESH_SECONDS:
                live_render(answer, slots)
                last = now
            if answer.done:
                break
    finish(answer, "Última resposta:", ss["messages"], "A")


def stream_worker(payload, out_queue):
    for event in stream_events(payload):
        out_queue.put(event)
    out_queue.put(None)


def run_compare(prompt):
    histories = {side: list(ss[f"cmp_{side}"]) for side in ("A", "B")}
    columns = dict(zip(("A", "B"), st.columns(2)))
    sequential = ss["compare_order"] == ORDER_SEQUENTIAL
    state = {}
    for side in ("A", "B"):
        ss[f"cmp_{side}"].append({"role": "user", "content": prompt})
        with columns[side]:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                slots = assistant_slots()
                if sequential and side == "B":
                    slots["answer"].markdown("_Aguardando o endpoint A terminar..._")
        state[side] = {"answer": Answer(), "slots": slots, "queue": queue.Queue(), "finished": False,
                       "started": False, "payload": build_payload(prompt, side, histories[side])}

    def start(side):
        s = state[side]
        threading.Thread(target=stream_worker, args=(s["payload"], s["queue"]), daemon=True).start()
        s["started"] = True

    start("A")
    if not sequential:
        start("B")

    while not all(s["finished"] for s in state.values()):
        if sequential and state["A"]["finished"] and not state["B"]["started"]:
            start("B")
        for s in state.values():
            changed = False
            while True:
                try:
                    event = s["queue"].get_nowait()
                except queue.Empty:
                    break
                if event is None:
                    s["finished"] = True
                    break
                s["answer"].apply(event)
                changed = True
            if changed:
                live_render(s["answer"], s["slots"])
        time.sleep(UI_REFRESH_SECONDS)

    for side in ("A", "B"):
        finish(state[side]["answer"], f"{side}:", ss[f"cmp_{side}"], side)


def compare_summary():
    """Compara as métricas da última rodada do modo comparação."""
    last = {}
    for side in ("A", "B"):
        msgs = [m for m in ss[f"cmp_{side}"] if m["role"] == "assistant"]
        if not msgs or not msgs[-1].get("metrics") or msgs[-1].get("error"):
            return
        last[side] = msgs[-1]["metrics"]
    parts = []
    a, b = last["A"].get("ttft"), last["B"].get("ttft")
    if a and b:
        fast, slow = ("A", "B") if a <= b else ("B", "A")
        ratio = max(a, b) / min(a, b)
        parts.append(f"⏱️ 1º token: **{fast}** foi {ratio:.1f}x mais rápido que {slow}")
    a, b = last["A"].get("tps"), last["B"].get("tps")
    if a and b:
        fast, slow = ("A", "B") if a >= b else ("B", "A")
        ratio = max(a, b) / min(a, b)
        parts.append(f"⚡ Geração: **{fast}** foi {ratio:.1f}x mais rápido que {slow} ({max(a, b)} vs {min(a, b)} tok/s)")
    if parts:
        st.info("**Última rodada** · " + " · ".join(parts))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

BENCH_METRICS = [
    # chave, rótulo, menor é melhor?, casas decimais
    ("ttft_ms", "Tempo até o 1º token (ms)", True, 1),
    ("e2e_ms", "Tempo total da resposta (ms)", True, 1),
    ("tpot_ms", "Tempo por token de saída (ms)", True, 2),
    ("decode_tps", "Velocidade de geração (tok/s)", False, 1),
    ("itl_ms", "Intervalo entre chunks (ms)", True, 2),
    ("output_tokens", "Tokens de saída", None, 0),
    ("prompt_tokens", "Tokens de prompt", None, 0),
]

METHODOLOGY = """
- **Onde é medido:** no backend, dentro do cluster, perto do NAI. O navegador e o Streamlit não entram na conta.
- **Relógio:** monotônico de alta resolução, lido no instante em que cada chunk chega.
- **Tempo até o 1º token (TTFT):** do momento em que a requisição termina de ser enviada até o primeiro
  chunk com texto (raciocínio ou resposta). Não inclui a abertura da conexão TCP/TLS.
- **Tempo por token (TPOT) e velocidade de geração:** (último token − primeiro token) ÷ (tokens − 1).
  É a velocidade de decode, sem o tempo de processar o prompt.
- **Tokens:** pelo campo `usage` do próprio NAI. Se o endpoint não informar, são estimados pelo número
  de chunks e aparecem marcados como estimados.
- **Aquecimento:** as primeiras requisições abrem as conexões e aquecem o modelo e não entram nas estatísticas.
  As conexões são reaproveitadas (keep-alive) nas requisições medidas.
- **Evitar cache de prompt:** cada requisição recebe um prefixo único, para o cache de prefixo do vLLM
  não deixar o TTFT artificialmente baixo.
- **Tamanho fixo de saída** (`ignore_eos`, parâmetro do vLLM): força todas as respostas a terem exatamente
  o máximo de tokens, deixando as rodadas comparáveis. Alguns gateways podem recusar o parâmetro.
- **Concorrência:** quantas requisições ficam em andamento ao mesmo tempo. Com 1, mede a latência de um
  usuário sozinho; com mais, mede como o endpoint se comporta sob carga (o throughput agregado soma todas).
- **Percentis:** p50 é a mediana; p90/p99 mostram a cauda. Com poucas requisições, p99 é pouco representativo.
- **Throughput em regime:** tokens gerados entre o instante em que todas as vagas de concorrência ficam
  ocupadas e o início da última requisição, dividido por esse intervalo. Exclui a subida e a descida da carga.
- **Linha do tempo:** cada chunk recebido conta como (tokens da requisição ÷ chunks), então a soma do gráfico
  bate com o total informado pelo NAI.
- **Varredura de concorrência:** repete a medição para cada nível, com (fator × concorrência) requisições,
  um nível depois do outro.
- **Varredura de contexto:** antes de cada tamanho, duas requisições de calibração medem quantos caracteres
  valem um token no tokenizador do próprio modelo; o prompt é montado para chegar ao tamanho pedido e o
  valor real (do `usage`) é o que aparece nos gráficos.
- **KV cache e fila:** lidos do `/metrics` do servidor de inferência (vLLM) a cada 0,5 s. O KV cache guarda
  o contexto das requisições em andamento na memória da GPU; perto de 100%, o servidor passa a enfileirar ou
  preemptar requisições. Picos mais curtos que o intervalo de amostragem podem não aparecer.
"""


def bench_body(side, **overrides):
    cfg = endpoint_config(side)
    body = {
        **cfg,
        "prompt": ss["bench_prompt"],
        "requests": int(ss["bench_n"]),
        "concurrency": int(ss["bench_conc"]),
        "warmup": int(ss["bench_warmup"]),
        "max_tokens": int(ss["bench_max_tokens"]),
        "temperature": float(ss["bench_temp"]),
        "avoid_prompt_cache": bool(ss["bench_avoid_cache"]),
        "ignore_eos": bool(ss["bench_ignore_eos"]),
        "metrics_url": ss[f"metrics_{side}"].strip() or None,
    }
    body.update(overrides)
    return body


def parse_levels(text, minimum, maximum):
    """'1, 2, 4' -> [1, 2, 4] (sem repetição, em ordem, dentro dos limites)."""
    levels = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip().lower().replace(".", "").replace("k", "000")
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise ValueError(f"'{part}' não é um número inteiro.")
        if not minimum <= value <= maximum:
            raise ValueError(f"{value} está fora do intervalo {minimum}–{maximum}.")
        if value not in levels:
            levels.append(value)
    if not levels:
        raise ValueError("Informe pelo menos um valor.")
    return sorted(levels)


def bench_jobs(side, kind):
    """Lista de (nível, corpo da requisição) que um endpoint vai executar."""
    if kind == KIND_CONC:
        mult = int(ss["bench_per_level"])
        return [(c, bench_body(side, concurrency=c, requests=max(4, mult * c)))
                for c in parse_levels(ss["bench_levels_conc"], 1, 64)]
    if kind == KIND_CTX:
        return [(n, bench_body(side, prompt_tokens_target=n))
                for n in parse_levels(ss["bench_levels_ctx"], 16, 1_000_000)]
    return [(None, bench_body(side))]


def bench_worker(body, out_queue):
    for event in stream_events(body, path="benchmark", read_timeout=1800):
        out_queue.put(event)
    out_queue.put(None)


def _median(values):
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2


def level_label(kind, level):
    if kind == KIND_CONC:
        return f"concorrência {level}"
    if kind == KIND_CTX:
        return f"contexto {level:,} tokens".replace(",", ".")
    return ""


def bench_progress_text(side, lane, kind):
    prefix = f"Endpoint {side}"
    if lane["started"] and lane["job"] < len(lane["jobs"]):
        level = lane["jobs"][lane["job"]][0]
        if level is not None:
            prefix += f" · {level_label(kind, level)} ({lane['job'] + 1}/{len(lane['jobs'])})"
    cur = lane["current"]
    if not lane["started"]:
        return f"{prefix}: aguardando"
    if lane["finished"]:
        failed = sum(1 for p in lane["points"] if p["error"])
        return f"Endpoint {side}: concluído" + (f" · {failed} nível(is) com erro" if failed else "")
    if cur["calibrating"]:
        return f"{prefix}: calibrando o tamanho do contexto"
    if not cur["results"]:
        return f"{prefix}: aquecendo ({cur['warmup_done']}/{cur['warmup_total']})"
    ok = [r for r in cur["results"] if r["ok"]]
    parts = [f"{prefix}: {len(cur['results'])}/{cur['total']}"]
    ttft = _median([r["ttft_ms"] for r in ok])
    tps = _median([r["decode_tps"] for r in ok])
    if ttft is not None:
        parts.append(f"1º token {ttft:.0f} ms")
    if tps is not None:
        parts.append(f"{tps:.1f} tok/s")
    failed = len(cur["results"]) - len(ok)
    if failed:
        parts.append(f"{failed} falha(s)")
    return " · ".join(parts)


def run_benchmark_ui(sides, kind):
    """Executa os níveis de cada endpoint em sequência. Endpoints em paralelo só
    na rodada única, quando escolhido; nas varreduras sempre um por vez."""
    sequential = len(sides) == 1 or kind != KIND_SINGLE or ss["bench_order"] == BENCH_SEQUENTIAL
    lanes = {}
    for side in sides:
        lanes[side] = {"jobs": bench_jobs(side, kind), "job": 0, "started": False, "finished": False,
                       "points": [], "queue": None, "current": None,
                       "bar": st.progress(0.0, text=f"Endpoint {side}: aguardando")}

    def new_current(body):
        return {"results": [], "summary": None, "error": None, "total": body["requests"],
                "warmup_total": body["warmup"], "warmup_done": 0, "calibrating": bool(body.get("prompt_tokens_target"))}

    def start_job(side):
        lane = lanes[side]
        level, body = lane["jobs"][lane["job"]]
        lane["queue"] = queue.Queue()
        lane["current"] = new_current(body)
        threading.Thread(target=bench_worker, args=(body, lane["queue"]), daemon=True).start()
        lane["started"] = True

    started_at = datetime.now()
    if sequential:
        start_job(sides[0])
    else:
        for side in sides:
            start_job(side)

    while not all(lane["finished"] for lane in lanes.values()):
        if sequential:
            for i, side in enumerate(sides[1:], start=1):
                if lanes[sides[i - 1]]["finished"] and not lanes[side]["started"]:
                    start_job(side)
        for side, lane in lanes.items():
            if not lane["started"] or lane["finished"]:
                continue
            changed = False
            while True:
                try:
                    event = lane["queue"].get_nowait()
                except queue.Empty:
                    break
                changed = True
                cur = lane["current"]
                if event is None:
                    level, body = lane["jobs"][lane["job"]]
                    lane["points"].append({
                        "level": level, "body": body,
                        "results": sorted(cur["results"], key=lambda r: r.get("index", 0)),
                        "summary": cur["summary"], "error": cur["error"],
                    })
                    lane["job"] += 1
                    if lane["job"] < len(lane["jobs"]):
                        start_job(side)
                    else:
                        lane["finished"] = True
                    break
                kind_ev = event.get("type")
                if kind_ev == "calibration":
                    cur["calibrating"] = False
                elif kind_ev == "warmup":
                    cur["calibrating"] = False
                    cur["warmup_done"] = event["done"]
                elif kind_ev == "result":
                    cur["calibrating"] = False
                    cur["results"].append(event["result"])
                elif kind_ev == "summary":
                    cur["summary"] = event["summary"]
                elif kind_ev == "error":
                    cur["error"] = event["message"]
            if changed:
                n = len(lane["jobs"])
                cur = lane["current"]
                inside = (len(cur["results"]) / cur["total"]) if cur and cur["total"] and not lane["finished"] else 0
                fraction = min(1.0, (lane["job"] + inside) / n)
                lane["bar"].progress(fraction, text=bench_progress_text(side, lane, kind))
        time.sleep(UI_REFRESH_SECONDS)

    for side, lane in lanes.items():
        if any(p["summary"] and p["summary"].get("succeeded") for p in lane["points"]):
            remember_endpoint(side)
    first_body = lanes[sides[0]]["jobs"][0][1]
    run = {
        "kind": kind,
        "timestamp": started_at.strftime("%d/%m/%Y %H:%M:%S"),
        "order": BENCH_SEQUENTIAL if sequential else BENCH_PARALLEL,
        "sides": {},
        "settings": {k: v for k, v in first_body.items()
                     if k not in ("endpoint_url", "api_key", "model_name", "profile", "metrics_url")},
    }
    if kind == KIND_CONC:
        run["settings"]["levels"] = [lvl for lvl, _ in lanes[sides[0]]["jobs"]]
        run["settings"]["per_level"] = int(ss["bench_per_level"])
    elif kind == KIND_CTX:
        run["settings"]["levels"] = [lvl for lvl, _ in lanes[sides[0]]["jobs"]]
    for side, lane in lanes.items():
        points = [{k: v for k, v in p.items() if k != "body"} for p in lane["points"]]
        entry = {"endpoint_url": lane["jobs"][0][1]["endpoint_url"],
                 "model_name": lane["jobs"][0][1]["model_name"], "points": points}
        if kind == KIND_SINGLE:
            entry.update({"results": points[0]["results"], "summary": points[0]["summary"],
                          "error": points[0]["error"]})
        else:
            errors = [p["error"] for p in points if p["error"]]
            entry["error"] = errors[0] if errors and len(errors) == len(points) else None
            entry["summary"] = None
            entry["results"] = []
        run["sides"][side] = entry
    ss["bench_run"] = run


def fmt_num(value, decimals=1):
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(value, sign=True):
    if value is None:
        return "—"
    return (f"{value:+.1f}%" if sign else f"{value:.1f}%").replace(".", ",")


def stats_table(summary):
    import pandas as pd

    rows = []
    for key, label, _, decimals in BENCH_METRICS:
        d = summary["metrics"].get(key)
        if not d:
            continue
        rows.append({
            "Métrica": label,
            "Média": fmt_num(d["mean"], decimals),
            "Desvio": fmt_num(d["stdev"], decimals),
            "CV": fmt_pct(d["cv"] * 100, sign=False) if d.get("cv") is not None else "—",
            "Mín": fmt_num(d["min"], decimals),
            "p50": fmt_num(d["p50"], decimals),
            "p90": fmt_num(d["p90"], decimals),
            "p99": fmt_num(d["p99"], decimals),
            "Máx": fmt_num(d["max"], decimals),
            "N": d["n"],
        })
    return pd.DataFrame(rows).set_index("Métrica") if rows else None


def p50(summary, key):
    d = (summary or {}).get("metrics", {}).get(key) if summary else None
    return d["p50"] if d else None


def metric_tiles(side, summary, reference=None):
    cols = st.columns(5)
    tiles = [
        ("1º token (p50)", p50(summary, "ttft_ms"), " ms", 0, True, "ttft_ms"),
        ("Geração (p50)", p50(summary, "decode_tps"), " tok/s", 1, False, "decode_tps"),
        ("Resposta completa (p50)", (p50(summary, "e2e_ms") or 0) / 1000 if p50(summary, "e2e_ms") else None,
         " s", 2, True, "e2e_ms"),
        ("Throughput agregado", summary.get("throughput_tps"), " tok/s", 1, False, None),
    ]
    for col, (label, value, unit, dec, lower_better, key) in zip(cols, tiles):
        delta = None
        if reference is not None and value is not None:
            ref = (reference.get("throughput_tps") if key is None else p50(reference, key))
            if ref and key == "e2e_ms":
                ref = ref / 1000
            if ref:
                delta = f"{fmt_pct((value / ref - 1) * 100)} vs A"
        col.metric(label, f"{fmt_num(value, dec)}{unit}" if value is not None else "—", delta=delta,
                   delta_color="inverse" if lower_better else "normal")
    cols[4].metric("Sucesso", f"{summary['succeeded']}/{summary['requests']}")


def comparison_table(run):
    import pandas as pd

    a, b = run["sides"]["A"]["summary"], run["sides"]["B"]["summary"]
    rows = []
    for key, label, lower_better, decimals in BENCH_METRICS:
        va, vb = p50(a, key), p50(b, key)
        if va is None or vb is None:
            continue
        diff = (vb / va - 1) * 100 if va else None
        if lower_better is None or diff is None or abs(diff) < 1:
            best = "≈"
        else:
            best = "A" if (va < vb) == lower_better else "B"
        rows.append({"Métrica (p50)": label, "A": fmt_num(va, decimals), "B": fmt_num(vb, decimals),
                     "B vs A": fmt_pct(diff), "Melhor": best})
    ta, tb = a.get("throughput_tps"), b.get("throughput_tps")
    if ta and tb:
        diff = (tb / ta - 1) * 100
        rows.append({"Métrica (p50)": "Throughput agregado (tok/s)", "A": fmt_num(ta), "B": fmt_num(tb),
                     "B vs A": fmt_pct(diff), "Melhor": "≈" if abs(diff) < 1 else ("A" if ta > tb else "B")})
    return pd.DataFrame(rows).set_index("Métrica (p50)") if rows else None


def per_request_chart(run, key, title):
    import pandas as pd

    series = {}
    for side, data in run["sides"].items():
        values = {r["index"]: r[key] for r in data["results"] if r.get("ok") and r.get(key) is not None}
        if values:
            series[f"Endpoint {side}"] = pd.Series(values)
    if series:
        st.markdown(f"**{title}**")
        df = pd.DataFrame(series)
        df.index.name = "Requisição"
        st.line_chart(df, height=220)


RAW_FIELDS = ("index", "ok", "start_s", "first_s", "end_s", "ttft_ms", "e2e_ms", "tpot_ms", "decode_tps",
              "output_tokens", "prompt_tokens", "estimated", "chunks", "error")


def bench_csv(run):
    import pandas as pd

    rows = []
    level_col = {KIND_CONC: "concorrencia", KIND_CTX: "contexto_alvo"}.get(run.get("kind"))
    for side, data in run["sides"].items():
        for point in data.get("points") or [{"level": None, "results": data["results"]}]:
            for r in point["results"]:
                row = {"endpoint": side, "modelo": data["model_name"]}
                if level_col:
                    row[level_col] = point["level"]
                row.update({k: r.get(k) for k in RAW_FIELDS})
                rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def df_markdown(df):
    """Tabela Markdown a partir de um DataFrame (sem depender do pacote tabulate)."""
    headers = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def server_lines(summary):
    """Resumo das métricas do servidor (KV cache etc.) em texto."""
    server = (summary or {}).get("server")
    if not server:
        return []
    if server.get("error") and not server.get("samples"):
        return [f"Métricas do servidor indisponíveis: {server['error']}"]
    parts = []
    if server.get("kv_peak") is not None:
        parts.append(f"KV cache: pico amostrado {fmt_num(server['kv_peak'])}% (média {fmt_num(server['kv_mean'])}%)")
    if server.get("running_peak") is not None:
        parts.append(f"pico em execução {fmt_num(server['running_peak'], 0)}")
    if server.get("waiting_peak") is not None:
        parts.append(f"pico na fila {fmt_num(server['waiting_peak'], 0)}")
    if server.get("prefix_hit_rate") is not None:
        parts.append(f"acerto no cache de prefixo {fmt_num(server['prefix_hit_rate'])}%")
    if server.get("preemptions") is not None:
        parts.append(f"preempções {fmt_num(server['preemptions'], 0)}")
    return ["Servidor: " + " · ".join(parts)] if parts else []


SWEEP_COLUMNS = [
    # coluna, rótulo, casas
    ("prompt_tokens", "Tokens de prompt (p50)", 0),
    ("ttft_p50", "1º token p50 (ms)", 1),
    ("ttft_p90", "1º token p90 (ms)", 1),
    ("e2e_p50", "Resposta p50 (ms)", 0),
    ("decode_p50", "Geração por usuário p50 (tok/s)", 1),
    ("throughput", "Throughput agregado (tok/s)", 1),
    ("prefill_tps", "Prefill (tok/s)", 0),
    ("kv_peak", "KV cache pico (%)", 1),
    ("waiting_peak", "Fila pico", 0),
    ("falhas", "Falhas", 0),
]


def sweep_table(df, side_label, kind):
    sub = df[df["endpoint"] == side_label].sort_values("nivel")
    if sub.empty:
        return None
    level_title = "Concorrência" if kind == KIND_CONC else "Contexto alvo (tokens)"
    out = {level_title: [fmt_num(v, 0) for v in sub["nivel"]]}
    for col, label, dec in SWEEP_COLUMNS:
        if kind == KIND_CONC and col in ("prefill_tps",):
            continue
        if col in sub and sub[col].notna().any():
            out[label] = [fmt_num(v, dec) if v == v and v is not None else "—" for v in sub[col]]
    import pandas as pd

    return pd.DataFrame(out).set_index(level_title)


def bench_markdown(run):
    st_ = run["settings"]
    kind = run.get("kind", KIND_SINGLE)
    lines = [
        "# Benchmark — NTNX BR AI Assistant", "",
        f"_Executado em {run['timestamp']}_ · {kind}", "",
        "## Configuração", "",
    ]
    if kind == KIND_CONC:
        lines.append(f"- Níveis de concorrência: {', '.join(str(v) for v in st_['levels'])} "
                     f"({st_['per_level']} × concorrência requisições por nível)")
    else:
        lines += [f"- Requisições medidas: {st_['requests']}", f"- Concorrência: {st_['concurrency']}"]
    if kind == KIND_CTX:
        lines.append(f"- Tamanhos de contexto (tokens): {', '.join(str(v) for v in st_['levels'])}")
    lines += [
        f"- Aquecimento: {st_['warmup']}",
        f"- Máximo de tokens: {st_['max_tokens']} · Temperatura: {st_['temperature']}",
        f"- Evitar cache de prompt: {'sim' if st_['avoid_prompt_cache'] else 'não'}",
        f"- Tamanho fixo de saída (ignore_eos): {'sim' if st_['ignore_eos'] else 'não'}",
    ]
    if len(run["sides"]) > 1:
        lines.append(f"- Execução: {run['order']}")
    lines += ["", "**Prompt:**", "", f"> {st_['prompt']}", ""]

    if kind != KIND_SINGLE:
        df = bc.sweep_rows({side: data["points"] for side, data in run["sides"].items()})
        for side, data in run["sides"].items():
            lines += [f"## Endpoint {side}", "", f"- URL: `{data['endpoint_url']}`", f"- Modelo: `{data['model_name']}`", ""]
            errors = [f"{level_label(kind, p['level'])}: {p['error']}" for p in data["points"] if p["error"]]
            lines += [f"> ⚠️ {e}" for e in errors] + ([""] if errors else [])
            table = sweep_table(df, f"Endpoint {side}", kind) if not df.empty else None
            if table is not None:
                lines += [df_markdown(table), ""]
        lines += ["## Metodologia", METHODOLOGY]
        return "\n".join(lines)

    for side, data in run["sides"].items():
        lines += [f"## Endpoint {side}", "", f"- URL: `{data['endpoint_url']}`", f"- Modelo: `{data['model_name']}`", ""]
        if data["error"]:
            lines += [f"> ⚠️ Erro: {data['error']}", ""]
            continue
        summary = data["summary"]
        if not summary:
            continue
        lines += [f"- Sucesso: {summary['succeeded']}/{summary['requests']}",
                  f"- Throughput agregado: {fmt_num(summary.get('throughput_tps'))} tok/s"
                  + (f" (em regime: {fmt_num(summary['steady_throughput_tps'])} tok/s)"
                     if summary.get("steady_throughput_tps") else ""),
                  f"- Duração total: {fmt_num(summary.get('wall_time_s'), 2)} s"]
        lines += [f"- {line}" for line in server_lines(summary)]
        if summary.get("estimated_tokens"):
            lines.append("- Tokens estimados (o endpoint não informou o uso)")
        lines.append("")
        table = stats_table(summary)
        if table is not None:
            lines += [df_markdown(table), ""]
    if len(run["sides"]) > 1 and all(d["summary"] for d in run["sides"].values()):
        table = comparison_table(run)
        if table is not None:
            lines += ["## Comparação", "", df_markdown(table), ""]
    lines += ["## Metodologia", METHODOLOGY]
    return "\n".join(lines)


def show_chart(chart, caption=None):
    if chart is None:
        return False
    st.altair_chart(bc.localized(chart), width="stretch")
    if caption:
        st.caption(caption)
    return True


def render_downloads(run):
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    col1, col2 = st.columns(2)
    col1.download_button("⬇️ Relatório (Markdown)", data=bench_markdown(run),
                         file_name=f"benchmark-nai-{stamp}.md", mime="text/markdown", width="stretch")
    col2.download_button("⬇️ Dados brutos (CSV)", data=bench_csv(run),
                         file_name=f"benchmark-nai-{stamp}.csv", mime="text/csv", width="stretch")


def render_detail_charts(run):
    """Gráficos detalhados de uma rodada única."""
    sides = run["sides"]
    results = {side: d["results"] for side, d in sides.items() if d["summary"]}
    summaries = {side: d["summary"] for side, d in sides.items() if d["summary"]}
    if not results:
        return
    has_server = any(((s or {}).get("server") or {}).get("samples") for s in summaries.values())
    names = ["⏱️ Linha do tempo", "📶 Concorrência", "🚀 Throughput no tempo"]
    if has_server:
        names.append("🧠 KV cache e fila")
    names += ["📊 Distribuições", "📈 Por requisição"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["⏱️ Linha do tempo"]:
        show_chart(bc.gantt_chart(results),
                   "Cada barra é uma requisição: laranja até o 1º token (fila + processamento do prompt), "
                   "azul durante a geração.")
    with tabs["📶 Concorrência"]:
        show_chart(bc.inflight_chart(results),
                   "Requisições em andamento a cada instante, separadas entre esperando o 1º token e gerando. "
                   "Muito laranja empilhado indica fila ou prefill demorado.")
        if has_server:
            show_chart(bc.server_queue_chart(summaries), "Visto pelo servidor de inferência (/metrics).")
    with tabs["🚀 Throughput no tempo"]:
        show_chart(bc.throughput_timeline_chart(summaries),
                   "Tokens gerados por segundo somando todas as requisições. A linha tracejada é o throughput "
                   "em regime (todas as vagas ocupadas), sem a subida e a descida da carga.")
    if has_server:
        with tabs["🧠 KV cache e fila"]:
            show_chart(bc.kv_cache_chart(summaries),
                       "Uso do KV cache do servidor (memória da GPU que guarda o contexto das requisições "
                       "em andamento), amostrado a cada 0,5 s.")
            show_chart(bc.server_queue_chart(summaries))
            for side, s in summaries.items():
                for line in server_lines(s):
                    st.caption(f"**Endpoint {side}** · {line}")
    with tabs["📊 Distribuições"]:
        col1, col2 = st.columns(2)
        with col1:
            show_chart(bc.histogram_chart(results, "ttft_ms", "Tempo até o 1º token (ms)"))
            show_chart(bc.histogram_chart(results, "e2e_ms", "Tempo total da resposta (ms)"))
        with col2:
            show_chart(bc.histogram_chart(results, "decode_tps", "Velocidade de geração (tok/s)"))
            show_chart(bc.histogram_chart(results, "output_tokens", "Tokens de saída"))
    with tabs["📈 Por requisição"]:
        col1, col2 = st.columns(2)
        with col1:
            per_request_chart(run, "ttft_ms", "Tempo até o 1º token por requisição (ms)")
        with col2:
            per_request_chart(run, "decode_tps", "Velocidade de geração por requisição (tok/s)")


def render_bench_results(run):
    st.divider()
    kind = run.get("kind", KIND_SINGLE)
    st.subheader(f"Resultado · {run['timestamp']}")
    if kind != KIND_SINGLE:
        render_sweep_results(run)
        return
    sides = run["sides"]
    two = len(sides) > 1
    if two:
        st.caption(f"Execução: {run['order']}")

    for side, data in sides.items():
        label = f"Endpoint {side}" if two else "Endpoint"
        st.markdown(f"**{label}** · `{data['model_name']}`")
        if data["error"]:
            st.error(data["error"])
            continue
        summary = data["summary"]
        if not summary:
            st.warning("O benchmark foi interrompido antes de terminar.")
            continue
        reference = sides["A"]["summary"] if (two and side == "B") else None
        metric_tiles(side, summary, reference)
        if summary.get("steady_throughput_tps"):
            st.caption(f"Throughput em regime (todas as {summary['settings']['concurrency']} vagas ocupadas): "
                       f"**{fmt_num(summary['steady_throughput_tps'])} tok/s**")
        for line in server_lines(summary):
            st.caption(line)
        if summary["failed"]:
            st.warning(f"{summary['failed']} requisição(ões) falharam: " + "; ".join(summary["errors"]))
        if summary.get("estimated_tokens"):
            st.caption("~ O endpoint não informou o uso de tokens: contagens e velocidades são estimadas pelo número de chunks.")
        if summary.get("streaming") is False:
            st.caption("O endpoint não respondeu em streaming: só o tempo total é medido.")

    if two and all(d["summary"] for d in sides.values()):
        table = comparison_table(run)
        if table is not None:
            st.markdown("**Comparação (medianas)**")
            st.dataframe(table, width="stretch")

    render_detail_charts(run)

    tabs = st.tabs([f"Estatísticas {s}" for s in sides] + ["Requisições"])
    for tab, (side, data) in zip(tabs, sides.items()):
        with tab:
            table = stats_table(data["summary"]) if data["summary"] else None
            if table is not None:
                st.dataframe(table, width="stretch")
                n = data["summary"]["succeeded"]
                if n < 50:
                    st.caption(f"Com {n} requisições, o p99 é pouco representativo; use p50 e p90.")
            else:
                st.write("Sem dados.")
    with tabs[-1]:
        import pandas as pd

        rows = []
        for side, data in sides.items():
            for r in data["results"]:
                rows.append({"Endpoint": side, "#": r.get("index"), "OK": "✅" if r["ok"] else "❌",
                             "Início (s)": r.get("start_s"), "1º token (ms)": r.get("ttft_ms"),
                             "Total (ms)": r.get("e2e_ms"), "tok/s": r.get("decode_tps"),
                             "Tokens": r.get("output_tokens"), "Erro": r.get("error") or ""})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    render_downloads(run)


def render_sweep_results(run):
    kind = run["kind"]
    sides = run["sides"]
    if len(sides) > 1:
        st.caption("Endpoints medidos um de cada vez, para um não interferir no outro.")
    for side, data in sides.items():
        failed = [p for p in data["points"] if p["error"]]
        for p in failed:
            st.warning(f"Endpoint {side} · {level_label(kind, p['level'])}: {p['error']}")
    df = bc.sweep_rows({side: data["points"] for side, data in sides.items()})
    if df.empty:
        st.error("Nenhum nível terminou com sucesso.")
        render_downloads(run)
        return

    if kind == KIND_CONC:
        c1, c2 = st.columns(2)
        with c1:
            show_chart(bc.sweep_line(df, "concorrencia", "Concorrência", [("throughput", "Throughput agregado")],
                                     "Throughput agregado (tok/s)", log_x=True),
                       "Tokens/s somando todas as requisições (em regime, quando disponível). Onde a curva "
                       "achata, o servidor saturou.")
            show_chart(bc.sweep_line(df, "concorrencia", "Concorrência", [("decode_p50", "p50")],
                                     "Geração por usuário (tok/s)", log_x=True),
                       "Velocidade que cada usuário percebe. Cai quando o lote fica cheio.")
        with c2:
            show_chart(bc.sweep_line(df, "concorrencia", "Concorrência", [("ttft_p50", "p50"), ("ttft_p90", "p90")],
                                     "Tempo até o 1º token (ms)", log_x=True),
                       "Sobe rápido quando as requisições começam a esperar na fila.")
            show_chart(bc.sweep_line(df, "concorrencia", "Concorrência", [("e2e_p50", "p50"), ("e2e_p90", "p90")],
                                     "Tempo total da resposta (ms)", log_x=True))
        show_chart(bc.latency_throughput_chart(df, "ttft_p50", "Tempo até o 1º token p50 (ms)"),
                   "Curva latência × throughput: cada ponto é um nível de concorrência. O ponto ideal de operação "
                   "fica antes da latência disparar.")
        if df["kv_peak"].notna().any() or df["waiting_peak"].notna().any():
            c1, c2 = st.columns(2)
            with c1:
                show_chart(bc.sweep_line(df, "concorrencia", "Concorrência", [("kv_peak", "pico")],
                                         "KV cache (%)", log_x=True), "Pico amostrado do uso do KV cache.")
            with c2:
                show_chart(bc.sweep_line(df, "concorrencia", "Concorrência", [("waiting_peak", "pico")],
                                         "Requisições na fila do servidor", log_x=True))
    else:
        c1, c2 = st.columns(2)
        with c1:
            show_chart(bc.sweep_line(df, "prompt_tokens", "Tokens de prompt", [("ttft_p50", "p50"), ("ttft_p90", "p90")],
                                     "Tempo até o 1º token (ms)", log_x=True),
                       "Quanto o tamanho do contexto pesa no tempo de processar o prompt (prefill).")
            show_chart(bc.sweep_line(df, "prompt_tokens", "Tokens de prompt", [("decode_p50", "p50")],
                                     "Geração por usuário (tok/s)", log_x=True),
                       "Contextos longos costumam deixar a geração mais lenta: cada token lê mais KV cache.")
        with c2:
            show_chart(bc.sweep_line(df, "prompt_tokens", "Tokens de prompt", [("prefill_tps", "p50")],
                                     "Velocidade de prefill (tokens de prompt/s)", log_x=True),
                       "Tokens de prompt divididos pelo tempo até o 1º token.")
            show_chart(bc.sweep_line(df, "prompt_tokens", "Tokens de prompt", [("e2e_p50", "p50")],
                                     "Tempo total da resposta (ms)", log_x=True))
        if df["kv_peak"].notna().any():
            show_chart(bc.sweep_line(df, "prompt_tokens", "Tokens de prompt", [("kv_peak", "pico")],
                                     "KV cache (%)", log_x=True),
                       "Quanto da memória de KV cache cada tamanho de contexto ocupa.")

    tabs = st.tabs([f"Tabela {side}" for side in sides])
    for tab, side in zip(tabs, sides):
        with tab:
            table = sweep_table(df, f"Endpoint {side}", kind)
            if table is not None:
                st.dataframe(table, width="stretch")
    render_downloads(run)


def render_benchmark():
    sides = ["A", "B"] if ss["bench_include_b"] else ["A"]
    st.markdown("Mede o desempenho do endpoint com estatística, repetindo a mesma pergunta. "
                "O system prompt, o histórico e o documento do chat não são usados aqui.")
    st.radio("Tipo de teste", [KIND_SINGLE, KIND_CONC, KIND_CTX], key="bench_kind", horizontal=True,
             help="Rodada única: uma configuração. Varredura de concorrência: repete com 1, 2, 4... usuários "
                  "simultâneos. Varredura de contexto: repete com prompts de tamanhos diferentes.")
    kind = ss["bench_kind"]
    with st.form("bench_form"):
        st.text_area("Prompt" if kind != KIND_CTX else "Pergunta (o contexto é gerado automaticamente)",
                     key="bench_prompt", height=100)
        if kind == KIND_CONC:
            c1, c2, c3, c4 = st.columns(4)
            c1.text_input("Níveis de concorrência", key="bench_levels_conc", help="Separados por vírgula, de 1 a 64.")
            c2.number_input("Requisições por nível (× concorrência)", min_value=1, max_value=20, step=1,
                            key="bench_per_level",
                            help="Com 4 e concorrência 8, o nível mede 32 requisições. Mais requisições, mais "
                                 "tempo em regime estável.")
        elif kind == KIND_CTX:
            c1, c2, c3, c4 = st.columns(4)
            c1.text_input("Tamanhos de contexto (tokens)", key="bench_levels_ctx",
                          help="Separados por vírgula. Aceita 4k, 8k... O tamanho é calibrado com o tokenizador "
                               "do próprio modelo.")
            c2.number_input("Requisições por tamanho", min_value=1, max_value=500, step=1, key="bench_n")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("Concorrência", min_value=1, max_value=64, step=1, key="bench_conc")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("Requisições medidas", min_value=1, max_value=500, step=1, key="bench_n")
            c2.number_input("Concorrência", min_value=1, max_value=64, step=1, key="bench_conc",
                            help="Requisições em andamento ao mesmo tempo. 1 = um usuário sozinho.")
        c3.number_input("Aquecimento", min_value=0, max_value=10, step=1, key="bench_warmup",
                        help="Requisições iniciais descartadas das estatísticas (em cada nível).")
        c4.number_input("Máximo de tokens", min_value=1, max_value=32768, step=64, key="bench_max_tokens")
        c1, c2, c3 = st.columns([1, 1.4, 1.6])
        c1.number_input("Temperatura", min_value=0.0, max_value=2.0, step=0.1, key="bench_temp")
        c2.checkbox("Evitar cache de prompt", key="bench_avoid_cache",
                    help="Prefixo único em cada requisição, para o cache de prefixo não mascarar o 1º token.")
        c3.checkbox("Tamanho fixo de saída (ignore_eos)", key="bench_ignore_eos",
                    help="Parâmetro do vLLM: toda resposta tem exatamente o máximo de tokens. "
                         "Deixa rodadas comparáveis; alguns gateways podem recusar.")
        if len(sides) > 1 and kind == KIND_SINGLE:
            st.radio("Execução dos endpoints", [BENCH_SEQUENTIAL, BENCH_PARALLEL], key="bench_order",
                     horizontal=True,
                     help="Um por vez evita que os dois disputem a mesma GPU/rede e distorçam as medições.")
        if not any(ss[f"metrics_{side}"].strip() for side in sides):
            st.caption("💡 Para ver uso do KV cache e fila do servidor, informe a URL de métricas do vLLM "
                       "na barra lateral.")
        submitted = st.form_submit_button("▶️ Iniciar benchmark", type="primary")

    with st.expander("ℹ️ Como as métricas são medidas"):
        st.markdown(METHODOLOGY)

    if submitted:
        error = None
        if not ss["bench_prompt"].strip():
            error = "Escreva um prompt para o benchmark."
        else:
            try:
                if kind == KIND_CONC:
                    parse_levels(ss["bench_levels_conc"], 1, 64)
                elif kind == KIND_CTX:
                    parse_levels(ss["bench_levels_ctx"], 16, 1_000_000)
            except ValueError as e:
                error = f"Níveis inválidos: {e}"
        if error:
            st.error(error)
        else:
            run_benchmark_ui(sides, kind)
            st.rerun()

    if ss["bench_run"]:
        render_bench_results(ss["bench_run"])


if ss["mode"] == MODE_BENCH:
    render_benchmark()
elif ss["mode"] == MODE_COMPARE:
    compare_summary()
    col_a, col_b = st.columns(2)
    for side, col in (("A", col_a), ("B", col_b)):
        with col:
            st.markdown(f"**Endpoint {side}** · `{ss[f'model_{side}']}`")
            render_messages(ss[f"cmp_{side}"])
    if prompt := st.chat_input("Pergunte algo aos dois endpoints..."):
        run_compare(prompt)
        st.rerun()
else:
    render_messages(ss["messages"])
    if prompt := st.chat_input("Pergunte algo..."):
        run_chat(prompt)
        st.rerun()
