import streamlit as st
import requests

st.title("🤖 Coral Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
if prompt := st.chat_input("Como posso ajudar?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chamada para a sua API que está rodando no K8s
    # (No K8s, o streamlit chamaria o 'app-service')
    try:
        response = requests.post("http://app-service/api/v1/chat/ask", json={"user_input": prompt})
        bot_res = response.json()["response"]
    except:
        bot_res = "Erro ao conectar com a API."

    with st.chat_message("assistant"):
        st.markdown(bot_res)
    st.session_state.messages.append({"role": "assistant", "content": bot_res})