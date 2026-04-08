import os
import httpx 
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

# Variáveis de fallback (vêm do Manifesto)
DEFAULT_ENDPOINT = os.getenv("NAI_ENDPOINT")
DEFAULT_KEY = os.getenv("NAI_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL_NAME")

class ChatMessage(BaseModel):
    user_input: str
    history: List[Dict[str, Any]] = []
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None

@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    # Lógica: Usa o que vem do Front, se for nulo, usa o Default do Sistema
    final_endpoint = message.endpoint_url or DEFAULT_ENDPOINT
    final_key = message.api_key or DEFAULT_KEY
    final_model = message.model_name or DEFAULT_MODEL

    headers = {
        "Authorization": f"Bearer {final_key}",
        "Content-Type": "application/json"
    }
    
    # Monta o histórico completo
    all_messages = message.history + [{"role": "user", "content": message.user_input}]
    
    payload = {
        "model": final_model,
        "messages": all_messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(
                final_endpoint, 
                json=payload, 
                headers=headers, 
                timeout=120.0
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data['choices'][0]['message']['content']
                return {"status": "success", "response": bot_response}
            else:
                return {"status": "error", "response": f"Erro NAI ({response.status_code}): {response.text}"}
        except Exception as e:
            return {"status": "error", "response": f"Falha no Backend: {str(e)}"}