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
MODEL_NAME = os.getenv("MODEL_NAME", "coral-endpoint")

# 2. Modelo atualizado para suportar o histórico (Persistência)
router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    # Lógica de Fallback: Prioridade para o que vem do Front, depois o Sistema
    final_endpoint = message.endpoint_url or os.getenv("NAI_ENDPOINT")
    final_key = message.api_key or os.getenv("NAI_API_KEY")
    final_model = message.model_name or os.getenv("MODEL_NAME")

    headers = {
        "Authorization": f"Bearer {final_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": final_model,
        "messages": message.history + [{"role": "user", "content": message.user_input}],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(final_endpoint, json=payload, headers=headers, timeout=120.0)
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data['choices'][0]['message']['content']
                return {"status": "success", "response": bot_response}
            else:
                return {"status": "error", "response": f"Erro NAI ({response.status_code}): {response.text}"}
        except Exception as e:
            return {"status": "error", "response": f"Falha interna no Backend: {str(e)}"}