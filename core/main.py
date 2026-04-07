from fastapi import FastAPI
from core.modules import chatbot

app = FastAPI(title="App Modular")

@app.get("/")
def read_root():
    return {"status": "online", "version": "1.0.0"}

# Integrando o módulo de chatbot
app.include_router(chatbot.router, prefix="/chat")