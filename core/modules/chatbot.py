import os
import httpx
import time 
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

# Fallbacks lidos do Manifesto Kubernetes
DEFAULT_ENDPOINT = os.getenv("NAI_ENDPOINT", "https://10-54-94-16.sslip.nutanixdemo.com/enterpriseai/v1/chat/completions")
DEFAULT_KEY = os.getenv("NAI_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "coral-endpoint")

class ChatMessage(BaseModel):
    user_input: str
    history: List[Dict[str, Any]] = []
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None

@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    # Lógica: Prioriza o que vem do usuário no Frontend, senão usa o padrão do Manifesto
    final_endpoint = message.endpoint_url or DEFAULT_ENDPOINT
    final_key = message.api_key or DEFAULT_KEY
    final_model = message.model_name or DEFAULT_MODEL

    headers = {
        "Authorization": f"Bearer {final_key}", 
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": final_model,
        "messages": message.history + [{"role": "user", "content": message.user_input}],
        "temperature": 0.7
    }

    start_time = time.perf_counter() 
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(final_endpoint, json=payload, headers=headers, timeout=120.0)
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data['choices'][0]['message']['content']
                usage = data.get("usage", {})
                
                # Cálculo de Tokens por Segundo (Métrica de performance de Hardware)
                completion_tokens = usage.get("completion_tokens", 0)
                tps = completion_tokens / duration if duration > 0 else 0

                return {
                    "status": "success", 
                    "response": bot_response,
                    "metrics": {
                        "total_tokens": usage.get("total_tokens", 0),
                        "tps": round(tps, 2),
                        "duration": round(duration, 2)
                    }
                }
            return {"status": "error", "response": f"Erro NAI ({response.status_code}): {response.text}"}
        except Exception as e:
            return {"status": "error", "response": f"Erro de conexão: {str(e)}"}