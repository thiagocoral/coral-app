from fastapi import FastAPI
from core.modules import chatbot # Importamos nosso primeiro módulo

app = FastAPI(title="Coral App Modular")

@app.get("/")
async def root():
    return {"message": "Coral App Online", "modules": ["chatbot"]}

# Registro do roteador do chatbot
app.include_router(chatbot.router, prefix="/api/v1/chat", tags=["AI"])