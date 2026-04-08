# chatbot.py
from pydantic import BaseModel
from typing import List, Dict

class ChatMessage(BaseModel):
    user_input: str
    history: List[Dict[str, str]] = [] # Aceita a lista de mensagens anteriores

@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    headers = {
        "Authorization": f"Bearer {NAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Aqui replicamos a lógica do seu código antigo: 
    # Pegamos o histórico e adicionamos a nova pergunta do usuário
    messages_to_send = message.history + [{"role": "user", "content": message.user_input}]
    
    payload = {
        "model": "llama3-8b", # Ou o nome do seu endpoint
        "messages": messages_to_send,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    async with httpx.AsyncClient(verify=False) as client:
        # Aumentamos o timeout para 120s pois o histórico longo demora mais
        response = await client.post(NAI_ENDPOINT, json=payload, headers=headers, timeout=120.0)
        # ... resto do processamento (retornando data['choices'][0]['message']['content'])