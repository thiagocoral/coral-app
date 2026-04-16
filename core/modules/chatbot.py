import os
import httpx
import time
import json
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Imports do MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

router = APIRouter()

DEFAULT_ENDPOINT = os.getenv("NAI_ENDPOINT", "https://10-54-94-16.sslip.nutanixdemo.com/enterpriseai/v1/chat/completions")
DEFAULT_KEY = os.getenv("NAI_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "coral-endpoint")

MCP_SERVER_PARAMS = StdioServerParameters(
    command="python3",
    args=["/app/mcp_server.py"],
)

class ChatMessage(BaseModel):
    user_input: str
    history: List[Dict[str, Any]] = []
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None

@router.post("/ask")
async def ask_chatbot(message: ChatMessage):
    final_endpoint = message.endpoint_url or DEFAULT_ENDPOINT
    final_key = message.api_key or DEFAULT_KEY
    final_model = message.model_name or DEFAULT_MODEL
    start_time = time.perf_counter()

    # --- 1. ROTEAMENTO DE INTENÇÃO (FILTRO) ---
    user_query = message.user_input.lower()
    tech_keywords = ["hora", "horário", "recursos", "memória", "cpu", "arquivos", "listar", "conectividade", "ping", "status"]
    is_tech_query = any(k in user_query for k in tech_keywords)

    async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Montagem dinâmica do contexto
            if is_tech_query:
                # MODO AGENTE: Com ferramentas e histórico completo
                mcp_tools_list = await session.list_tools()
                nai_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    } for tool in mcp_tools_list.tools
                ]
                final_history = message.history
                tool_choice = "auto"
            else:
                # MODO CHAT: Sem ferramentas e sem histórico (limpa o vício)
                nai_tools = None 
                final_history = [] 
                tool_choice = None

            payload = {
                "model": final_model,
                "messages": [
                    {"role": "system", "content": "Você é o assistente NTNX BR. Responda apenas com texto amigável."},
                    *final_history,
                    {"role": "user", "content": message.user_input}
                ],
                "temperature": 0.0, # Máxima precisão
                "tools": nai_tools,
                "tool_choice": tool_choice
            }

            # Remove chaves None para evitar erro na API do Nutanix
            payload = {k: v for k, v in payload.items() if v is not None}

            async with httpx.AsyncClient(verify=False) as client:
                try:
                    headers = {"Authorization": f"Bearer {final_key}", "Content-Type": "application/json"}
                    response = await client.post(final_endpoint, json=payload, headers=headers, timeout=120.0)
                    
                    if response.status_code != 200:
                        return {"status": "error", "response": f"Erro NAI: {response.text}"}

                    data = response.json()
                    choice = data['choices'][0]['message']

                    # --- 2. PROCESSAMENTO DA RESPOSTA ---
                    if choice.get("tool_calls") and is_tech_query:
                        tool_call = choice["tool_calls"][0]
                        tool_name = tool_call["function"]["name"]
                        args_raw = tool_call["function"]["arguments"]
                        tool_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        
                        tool_result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        resultado = tool_result.content[0].text if hasattr(tool_result, 'content') else str(tool_result)
                        bot_response = f"🛠️ **Ação do Agente:** {tool_name}\n\n✅ **Resultado:** {resultado}"
                    else:
                        bot_response = choice.get('content', "Olá! Como posso ajudar?")

                    # Métricas
                    duration = time.perf_counter() - start_time
                    usage = data.get("usage", {})
                    tps = usage.get("completion_tokens", 0) / duration if duration > 0 else 0

                    return {
                        "status": "success", 
                        "response": bot_response,
                        "metrics": {"total_tokens": usage.get("total_tokens", 0), "tps": round(tps, 2), "duration": round(duration, 2)}
                    }
                except Exception as e:
                    return {"status": "error", "response": f"Erro: {str(e)}"}