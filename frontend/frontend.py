import streamlit as st
import requests
import time

# 1. Configuração da Página
st.set_page_config(page_title="Coral AI - Powered by Nutanix NAI", page_icon="🤖")
# Inicializa contadores globais na sessão se não existirem
if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0
# 2. Estilização Customizada
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Barra Lateral (Sidebar) - Configurações Dinâmicas
with st.sidebar:
    st.title("⚙️ Configurações de IA")
    st.info("Valores carregados automaticamente do Kubernetes.")
    
    # Buscamos do manifesto usando os.getenv
    # O segundo parâmetro é o "fallback" caso a variável não exista no pod
    user_endpoint = st.text_input(
        "Nutanix Endpoint URL", 
        value=os.getenv("NAI_ENDPOINT", "")
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
    col1, col2 = st.columns(2)
    col1.metric("Tokens Total", st.session_state.total_tokens_used)
    
    # Placeholder para métricas da última resposta
    last_lat = st.empty()
    last_tokens = st.empty()

# 4. Cabeçalho Principal
st.title("🤖 Coral AI Assistant")
st.caption("🚀 Rodando no NKP via **Nutanix Enterprise AI (NAI)**")

# URL fixa do seu serviço Backend no Kubernetes
API_URL = "http://app-service/api/v1/chat/ask"

# 5. Inicialização e Exibição do Histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Lógica de Interação (Chat Input)
if prompt := st.chat_input("Pergunte algo..."):
    # Adiciona e mostra a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Processamento da Resposta do Assistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Pensando...*")
        full_response = "Não foi possível obter resposta." 
        
        try:
            # Montamos o payload com o histórico E os dados da Sidebar
            payload = {
                "user_input": prompt,
                "history": st.session_state.messages[:-1], # Histórico acumulado
                "endpoint_url": user_endpoint,             # Vem da Sidebar
                "api_key": user_api_key,                   # Vem da Sidebar
                "model_name": user_model                   # Vem da Sidebar
            }
            
            # Chamada ao seu Backend FastAPI
            start_time = time.time()
            response = requests.post(API_URL, json=payload, timeout=90)
            end_time = time.time()
            
            if response.status_code == 200:
                res_json = response.json()
                full_response = res_json.get("response", "Erro no formato da resposta.")
                
                metrics = res_json.get("metrics", {})

                # Atualiza métricas globais
                st.session_state.total_tokens_used += metrics.get("total_tokens", 0)
                
                # Exibe métricas da última interação
                last_lat.caption(f"⏱️ Latência: {end_time - start_time:.2f}s")
                last_tokens.caption(f"🪙 Tokens nesta volta: {metrics.get('total_tokens')}")
                if res_json.get("status") == "error":
                    st.warning(full_response)
                else:
                    # Efeito de digitação para a resposta
                    displayed_text = ""
                    for char in full_response:
                        displayed_text += char
                        message_placeholder.markdown(displayed_text + "▌")
                        time.sleep(0.01)
                    message_placeholder.markdown(full_response)
                    
                    # Exibe a latência na Sidebar para dar um toque profissional
                    st.sidebar.metric("Latência NAI", f"{end_time - start_time:.2f}s")
            else:
                full_response = f"Erro na API ({response.status_code}): {response.text}"
                st.error(full_response)
                
        except Exception as e:
            full_response = f"Falha de conexão: {e}"
            st.error(full_response)

        # Adiciona a resposta final ao histórico de mensagens
        st.session_state.messages.append({"role": "assistant", "content": full_response})