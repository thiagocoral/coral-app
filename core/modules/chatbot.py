import os
import httpx # Recomendado para FastAPI por ser assíncrono
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Lendo das variáveis de ambiente do sistema
NAI_ENDPOINT = os.getenv("NAI_ENDPOINT", "https://10-54-94-16.sslip.nutanixdemo.com/enterpriseai/v1/chat/completions")
NAI_API_KEY = os.getenv("NAI_API_KEY")

class ChatMessage(BaseModel):
    user_input: str

@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    headers = {
        "Authorization": f"Bearer {NAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "coral-endpoint", # <--- CERTIFIQUE-SE QUE ESTE NOME ESTÁ IGUAL NO PAINEL DO NAI
        "messages": [{"role": "user", "content": message.user_input}],
        "temperature": 0.7
    }

    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.post(NAI_ENDPOINT, json=payload, headers=headers, timeout=120.0)
            
            if response.status_code == 200:
                data = response.json()
                # O NAI segue o padrão OpenAI: choices[0].message.content
                bot_response = data['choices'][0]['message']['content']
                return {"status": "success", "response": bot_response}
            else:
                # Retorna o erro real do NAI para facilitar o seu debug
                return {"status": "error", "response": f"Erro NAI ({response.status_code}): {response.text}"}
        except Exception as e:
            return {"status": "error", "response": f"Falha interna no Backend: {str(e)}"}