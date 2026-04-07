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
    
    # Estrutura padrão OpenAI que o NAI costuma seguir
    payload = {
        "model": "coral", # Nome da sua Key/Modelo no NAI
        "messages": [{"role": "user", "content": message.user_input}],
        "temperature": 0.7
    }

    async with httpx.AsyncClient(verify=False) as client: # verify=False se o SSL for self-signed
        response = await client.post(NAI_ENDPOINT, json=payload, headers=headers)
        
    if response.status_code == 200:
        data = response.json()
        # Ajuste o parsing conforme o retorno real do seu NAI
        bot_response = data['choices'][0]['message']['content']
        return {"status": "success", "response": bot_response}
    else:
        return {"status": "error", "message": f"Erro no NAI: {response.text}"}