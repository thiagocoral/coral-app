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

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://app-service").rstrip("/")
API = f"{BACKEND_URL}/api/v1/chat"
UI_REFRESH_SECONDS = 0.05
MODE_CHAT = "💬 Chat"
MODE_COMPARE = "⚖️ Comparação"
DOC_TYPES = ["pdf", "txt", "md", "csv", "json", "log", "yaml", "yml"]

FALLBACK_DEFAULTS = {
    "endpoint_url": "https://10.42.163.114/enterpriseai/gateway/v1/chat/completions",
    "model_name": "unified-endpoint",
    "system_prompt": "Você é o assistente NTNX BR, rodando no Nutanix Enterprise AI (NAI). "
                     "Responda de forma clara e objetiva, em português do Brasil.",
    "temperature": 0.3,
    "max_tokens": 2048,
    "doc_max_chars": 20000,
}

st.set_page_config(
    page_title="NTNX BR AI - Powered by Nutanix NAI and NKP",
    page_icon="🤖",
    layout="wide" if st.session_state.get("mode") == MODE_COMPARE else "centered",
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


def post_json(path, body, timeout=30):
    try:
        resp = requests.post(f"{API}/{path}", json=body, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Backend respondeu HTTP {resp.status_code}: {resp.text[:300]}"}
    except requests.RequestException as e:
        return {"error": f"Não foi possível falar com o backend ({BACKEND_URL}): {e}"}


def stream_events(payload):
    """Lê o NDJSON do backend e entrega um evento (dict) por vez."""
    try:
        with requests.post(f"{API}/stream", json=payload, stream=True, timeout=(10, 600)) as resp:
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
        "key_A": os.getenv("NAI_API_KEY", ""),
        "model_A": model,
        "endpoint_B": os.getenv("NAI_ENDPOINT_B") or endpoint,
        "key_B": os.getenv("NAI_API_KEY_B", ""),
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
ss = st.session_state


def endpoint_config(side):
    key = ss[f"key_{side}"] or (ss["key_A"] if side == "B" else "")
    return {
        "endpoint_url": ss[f"endpoint_{side}"],
        "api_key": key,
        "model_name": ss[f"model_{side}"],
    }


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


def endpoint_controls(side):
    st.text_input("URL do endpoint", key=f"endpoint_{side}")
    help_key = "Em branco: usa a mesma do endpoint A." if side == "B" else None
    st.text_input("API Key", key=f"key_{side}", type="password", help=help_key)

    models = ss[f"models_{side}"]
    if models:
        current = ss[f"model_{side}"]
        st.selectbox(
            "Modelos disponíveis",
            models,
            index=models.index(current) if current in models else None,
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
            result = post_json("models", {"endpoint_url": cfg["endpoint_url"], "api_key": cfg["api_key"]})
        if result.get("status") == "success":
            ss[f"models_{side}"] = result["models"]
            ss[f"conn_{side}"] = {"kind": "success", "text": f"{len(result['models'])} modelo(s) encontrado(s)."}
        else:
            ss[f"conn_{side}"] = {"kind": "warning",
                                  "text": result.get("message") or result.get("error") or "Falha ao listar modelos."}
        st.rerun()
    if col2.button("🩺 Testar", key=f"test_{side}", help="Testar rede, TLS e API Key", width="stretch"):
        with st.spinner("Testando conexão..."):
            cfg = endpoint_config(side)
            result = post_json("connectivity", {"endpoint_url": cfg["endpoint_url"], "api_key": cfg["api_key"]})
        if "error" in result:
            ss[f"conn_{side}"] = {"kind": "error", "text": result["error"]}
        else:
            kind = "success" if result.get("ok") else ("warning" if result.get("reachable") else "error")
            ss[f"conn_{side}"] = {"kind": kind, "text": result.get("message", "")}

    status = ss[f"conn_{side}"]
    if status:
        getattr(st, status["kind"])(status["text"])


with st.sidebar:
    st.title("⚙️ Configurações de IA")
    defaults, backend_online = backend_defaults(API)
    if not backend_online:
        st.warning(f"Backend indisponível em {BACKEND_URL}.")

    st.radio("Modo", [MODE_CHAT, MODE_COMPARE], key="mode", horizontal=True)
    comparing = ss["mode"] == MODE_COMPARE

    with st.expander("🔌 Endpoint A" if comparing else "🔌 Endpoint NAI", expanded=True):
        st.caption("Valores iniciais carregados do Kubernetes.")
        endpoint_controls("A")
    if comparing:
        with st.expander("🔌 Endpoint B", expanded=True):
            endpoint_controls("B")

    with st.expander("🎛️ Parâmetros do modelo"):
        st.text_area("System prompt", key="system_prompt", height=140)
        st.slider("Temperatura", 0.0, 2.0, step=0.05, key="temperature",
                  help="Mais baixo: respostas mais previsíveis. Mais alto: mais criativas.")
        st.number_input("Máximo de tokens na resposta", min_value=16, max_value=131072, step=256, key="max_tokens")

    with st.expander("📎 Documento", expanded=bool(ss["doc"])):
        uploaded = st.file_uploader("Pergunte sobre um arquivo", type=DOC_TYPES, key="uploader")
        if uploaded is None:
            ss["doc"] = None
        elif not ss["doc"] or ss["doc"]["name"] != uploaded.name or ss["doc"].get("size") != uploaded.size:
            try:
                doc = extract_document(uploaded)
                doc["size"] = uploaded.size
                ss["doc"] = doc
            except Exception as e:  # noqa: BLE001
                ss["doc"] = None
                st.error(f"Não foi possível ler o arquivo: {e}")
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
    st.metric("Tokens acumulados na sessão", f"{ss['total_tokens']:,}")
    for label, m in ss["last_metrics"].items():
        st.caption(f"**{label}** {fmt_metrics(m)}")


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------

st.title("🤖 NTNX BR AI Assistant")
st.caption("🚀 Rodando no NKP via **Nutanix Enterprise AI (NAI)**")
if ss.get("pending_toast"):
    st.toast(ss.pop("pending_toast"), icon="📎")
if ss["doc"] and ss["doc"]["text"]:
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


def finish(answer, label, store):
    if answer.notice:
        ss["pending_toast"] = answer.notice
    msg = answer.as_message()
    store.append(msg)
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
    finish(answer, "Última resposta:", ss["messages"])


def stream_worker(payload, out_queue):
    for event in stream_events(payload):
        out_queue.put(event)
    out_queue.put(None)


def run_compare(prompt):
    histories = {side: list(ss[f"cmp_{side}"]) for side in ("A", "B")}
    columns = dict(zip(("A", "B"), st.columns(2)))
    state = {}
    for side in ("A", "B"):
        ss[f"cmp_{side}"].append({"role": "user", "content": prompt})
        with columns[side]:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                slots = assistant_slots()
        events = queue.Queue()
        thread = threading.Thread(target=stream_worker, args=(build_payload(prompt, side, histories[side]), events),
                                  daemon=True)
        thread.start()
        state[side] = {"answer": Answer(), "slots": slots, "queue": events, "finished": False}

    while not all(s["finished"] for s in state.values()):
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
        finish(state[side]["answer"], f"{side}:", ss[f"cmp_{side}"])


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


if ss["mode"] == MODE_COMPARE:
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
