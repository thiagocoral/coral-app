from fastapi import APIRouter

router = APIRouter()

@router.post("/send")
async def ask_bot(message: str):
    # Aqui você conectará com OpenAI, LangChain, etc.
    return {"response": f"Você disse: {message}. Eu sou um módulo!"}