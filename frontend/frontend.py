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
st.caption("🚀 Rodando localmente via **Nutanix Enterprise AI (NAI)**")

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
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Pensando...*")
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json={"user_input": prompt}, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                full_response = response.json().get("response", "Erro na resposta.")
                # Simula efeito de digitação
                displayed_text = ""
                for char in full_response:
                    displayed_text += char
                    message_placeholder.markdown(displayed_text + "▌")
                    time.sleep(0.01)
                message_placeholder.markdown(full_response)
                
                # Barra lateral com métricas de performance (Valor do NAI!)
                st.sidebar.metric("Latência NAI", f"{end_time - start_time:.2f}s")
                st.sidebar.success("Dados Processados Localmente 🔒")
            else:
                st.error(f"Erro na API: {response.status_code}")
        except Exception as e:
            st.error(f"Falha de conexão: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})