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
if prompt := st.chat_input("Pergunte algo para a IA do Nutanix..."):
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # TODO O BLOCO ABAIXO DEVE ESTAR IDENTADO DENTRO DO "IF PROMPT"
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*Pensando...*")
        
        full_response = "Não foi possível obter resposta." 

        try:
            start_time = time.time()
            # Enviando o JSON que o FastAPI espera: {"user_input": "texto"}
            payload = {
                "user_input": prompt,
                "history": st.session_state.messages, # Enviamos o que já foi conversado
                "max_tokens": 200
            }
            
            response = requests.post(API_URL, json=payload, timeout=120)
            end_time = time.time()
            
            if response.status_code == 200:
                res_json = response.json()
                full_response = res_json.get("response", "Erro no formato da resposta.")
                
                if res_json.get("status") == "error":
                    st.warning(full_response)
                else:
                    displayed_text = ""
                    for char in full_response:
                        displayed_text += char
                        message_placeholder.markdown(displayed_text + "▌")
                        time.sleep(0.01)
                    message_placeholder.markdown(full_response)
            else:
                # Se der 422, vamos mostrar o que o FastAPI reclamou
                full_response = f"Erro na API {response.status_code}: {response.text}"
                st.error(full_response)

        except Exception as e:
            full_response = f"Falha de conexão: {e}"
            st.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})