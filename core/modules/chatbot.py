from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ChatMessage(BaseModel):
    user_input: str

@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    # Aqui entrará a lógica de IA. Por enquanto, uma resposta mockada:
    user_text = message.user_input.lower()
    
    if "olá" in user_text or "oi" in user_text:
        response = "Olá! Eu sou o módulo de Chatbot do Coral App rodando no NKP."
    else:
        response = f"Recebi sua mensagem: '{message.user_input}'. Estou pronto para ser integrado com LLMs!"
        
    return {"status": "success", "response": response}