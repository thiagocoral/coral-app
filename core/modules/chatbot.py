import os
import httpx 
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

# 1. O router PRECISA vir antes do @router.post
router = APIRouter()

# Lendo das variáveis de ambiente
NAI_ENDPOINT = os.getenv("NAI_ENDPOINT", "https://10-54-94-16.sslip.nutanixdemo.com/enterpriseai/v1/chat/completions")
NAI_API_KEY = os.getenv("NAI_API_KEY")

# 2. Modelo atualizado para suportar o histórico (Persistência)
class ChatMessage(BaseModel):
    user_input: str
    history: List[Dict[str, Any]] = []

@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    headers = {
        "Authorization": f"Bearer {NAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 3. Monta o payload enviando o histórico completo para o NAI
    messages_to_send = message.history + [{"role": "user", "content": message.user_input}]
    
    payload = {
        "model": "coral-endpoint", 
        "messages": messages_to_send,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    async with httpx.AsyncClient(verify=False) as client:
        try:
            # Timeout de 120s para aguentar o processamento do histórico
            response = await client.post(NAI_ENDPOINT, json=payload, headers=headers, timeout=120.0)
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data['choices'][0]['message']['content']
                return {"status": "success", "response": bot_response}
            else:
                return {"status": "error", "response": f"Erro NAI ({response.status_code}): {response.text}"}
        except Exception as e:
            return {"status": "error", "response": f"Falha interna no Backend: {str(e)}"}