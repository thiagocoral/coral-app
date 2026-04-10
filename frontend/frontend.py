import streamlit as st
import requests
import time
import os  # Adicionado para ler as variáveis do manifesto

# 1. Configuração da Página
st.set_page_config(page_title="NTNX BR AI - Powered by Nutanix NAI and NKP", page_icon="🤖")

# Inicializa contadores globais na sessão
if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0

# 2. Estilização Customizada
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Barra Lateral (Sidebar) - Configurações Dinâmicas e Métricas
with st.sidebar:
    st.title("⚙️ Configurações de IA")
    st.info("Valores carregados automaticamente do Kubernetes.")
    
    # Buscamos do manifesto usando os.getenv para preencher os campos
    user_endpoint = st.text_input(
        "Nutanix Endpoint URL", 
        value=os.getenv("NAI_ENDPOINT", "https://10-54-94-16.sslip.nutanixdemo.com/enterpriseai/v1/chat/completions")
    )
    user_model = st.text_input(
        "Model Name", 
        value=os.getenv("MODEL_NAME", "coral-endpoint")
    )
    user_api_key = st.text_input(
        "API Key", 
        value=os.getenv("NAI_API_KEY", ""), 
        type="password"
    )
    
    st.divider()
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("📊 Métricas de Consumo")
    st.metric("Acumulado de Tokens", f"{st.session_state.total_tokens_used:,}")
    
    # Placeholders para métricas em tempo real da última resposta
    last_lat = st.empty()
    last_tps = st.empty()

# 4. Cabeçalho Principal
st.title("🤖 NTNX BR AI Assistant")
st.caption("🚀 Rodando no NKP via **Nutanix Enterprise AI (NAI)**")

API_URL = "http://app-service/api/v1/chat/ask"

# 5. Histórico de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Lógica de Interação
if prompt := st.chat_input("Pergunte algo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Pensando...*")
        full_response = ""
        
        try:
            payload = {
                "user_input": prompt,
                "history": st.session_state.messages[:-1],
                "endpoint_url": user_endpoint,
                "api_key": user_api_key,
                "model_name": user_model
            }
            
            response = requests.post(API_URL, json=payload, timeout=95)
            
            if response.status_code == 200:
                res_json = response.json()
                full_response = res_json.get("response", "")
                metrics = res_json.get("metrics", {})

                # Atualiza métricas globais e locais
                st.session_state.total_tokens_used += metrics.get("total_tokens", 0)
                
                # Exibe métricas de performance na Sidebar
                last_lat.caption(f"⏱️ Latência: {metrics.get('duration')}s")
                last_tps.caption(f"⚡ Velocidade: {metrics.get('tps')} tokens/sec")

                if res_json.get("status") == "error":
                    st.warning(full_response)
                else:
                    # Efeito de digitação
                    displayed_text = ""
                    for char in full_response:
                        displayed_text += char
                        message_placeholder.markdown(displayed_text + "▌")
                        time.sleep(0.005)
                    message_placeholder.markdown(full_response)
            else:
                st.error(f"Erro na API: {response.text}")
        except Exception as e:
            st.error(f"Falha de conexão: {e}")

        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})