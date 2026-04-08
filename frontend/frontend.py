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
# Input do Usuário
# frontend.py
if prompt := st.chat_input("Pergunte algo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Replicando a lógica de enviar o histórico
            payload = {
                "user_input": prompt,
                "history": st.session_state.messages[:-1] # Envia tudo menos a última (que já vamos add no backend)
            }
            
            response = requests.post(API_URL, json=payload, timeout=90)
            
            if response.status_code == 200:
                full_response = response.json().get("response")
                # Exibe a resposta (com seu efeito de digitação)
                message_placeholder.markdown(full_response)
            else:
                st.error(f"Erro: {response.status_code}")
        except Exception as e:
            st.error(f"Erro de conexão: {e}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})