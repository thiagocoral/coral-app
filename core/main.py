from fastapi import FastAPI

from core.modules import chatbot

app = FastAPI(title="Coral App Modular")


@app.get("/")
async def root():
    return {"message": "Coral App Online", "modules": ["chatbot"]}


@app.get("/health")
async def health():
    """Usado pelos probes do Kubernetes. Não depende do NAI estar no ar."""
    return {"status": "ok"}


app.include_router(chatbot.router, prefix="/api/v1/chat", tags=["AI"])
