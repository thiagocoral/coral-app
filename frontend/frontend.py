import streamlit as st
import requests
import time

# Configuração da Página
st.set_page_config(page_title="Coral AI - Powered by Nutanix NAI", page_icon="🤖")

# Estilização Customizada
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Coral AI Assistant")
st.caption("🚀 Rodando no NKP via **Nutanix Enterprise AI (NAI)**")

# URL da sua API no Kubernetes (IP que você já tem)
API_URL = "http://app-service/api/v1/chat/ask"

# Inicializa o histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do Usuário
if prompt := st.chat_input("Perunte algo para a IA do Nutanix..."):
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chamada para o Backend (FastAPI)
    # No seu frontend.py, dentro do bloco de input:

with st.chat_message("assistant"):
    message_placeholder = st.empty()
    message_placeholder.markdown("*Pensando...*")
    
    # 1. INICIALIZE A VARIÁVEL AQUI
    full_response = "Desculpe, não consegui obter uma resposta a tempo." 

    try:
        start_time = time.time()
        # 2. AUMENTE O TIMEOUT PARA 60 SEGUNDOS
        response = requests.post(API_URL, json={"user_input": prompt}, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            res_json = response.json()
            full_response = res_json.get("response", "Erro no formato da resposta.")
            
            if res_json.get("status") == "error":
                st.warning(full_response)
            else:
                # Lógica de exibição (efeito de digitação) que você já tem...
                displayed_text = ""
                for char in full_response:
                    displayed_text += char
                    message_placeholder.markdown(displayed_text + "▌")
                    time.sleep(0.01)
                message_placeholder.markdown(full_response)
        else:
            full_response = f"Erro na API: Status {response.status_code}"
            st.error(full_response)

    except requests.exceptions.Timeout:
        full_response = "A IA demorou demais para responder (Timeout). Tente novamente."
        st.error(full_response)
    except Exception as e:
        full_response = f"Falha de conexão: {str(e)}"
        st.error(full_response)

    # Agora esta linha NUNCA vai dar NameError
    st.session_state.messages.append({"role": "assistant", "content": full_response})