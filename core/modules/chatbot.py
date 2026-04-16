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

# Configurações do Manifesto Kubernetes
DEFAULT_ENDPOINT = os.getenv("NAI_ENDPOINT", "https://10-54-94-16.sslip.nutanixdemo.com/enterpriseai/v1/chat/completions")
DEFAULT_KEY = os.getenv("NAI_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "coral-endpoint")

# Parâmetros para iniciar o servidor MCP (Localizado na raiz do projeto)
# Como este arquivo está em core/modules/, subimos dois níveis para achar o mcp_server.py
MCP_SERVER_PARAMS = StdioServerParameters(
    command="python3",
    args=["../../mcp_server.py"],
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

    # Inicia a sessão com o Servidor MCP via STDIO
    async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            # Inicializa a conexão com o servidor MCP
            await session.initialize()
            
            # 1. Busca ferramentas disponíveis no MCP Server
            mcp_tools_list = await session.list_tools()
            
            # Converte para o formato que o Nutanix NAI aceita (JSON Schema)
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

            headers = {
                "Authorization": f"Bearer {final_key}", 
                "Content-Type": "application/json"
            }
            
            # Monta o histórico com a nova mensagem
            current_messages = message.history + [{"role": "user", "content": message.user_input}]

            payload = {
                "model": final_model,
                "messages": current_messages,
                "temperature": 0.7,
                "tools": nai_tools,      # Envia as ferramentas para a IA
                "tool_choice": "auto"    # IA decide se precisa usar ferramenta
            }

            async with httpx.AsyncClient(verify=False) as client:
                try:
                    response = await client.post(final_endpoint, json=payload, headers=headers, timeout=120.0)
                    
                    if response.status_code != 200:
                        return {"status": "error", "response": f"Erro NAI: {response.text}"}

                    data = response.json()
                    choice = data['choices'][0]['message']

                    # 2. Lógica de Agente: O modelo decidiu chamar uma ferramenta?
                    if choice.get("tool_calls"):
                        for tool_call in choice["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            tool_args = json.loads(tool_call["function"]["arguments"])
                            
                            # Executa a ferramenta no MCP Server
                            tool_result = await session.call_tool(tool_name, arguments=tool_args)
                            
                            # Enriquecemos o bot_response com o retorno da ferramenta
                            # Em um fluxo completo, enviaríamos de volta ao NAI, 
                            # mas aqui retornamos direto para o usuário ver a ação do agente.
                            bot_response = f"[Agente Coral] Executando {tool_name}...\nResultado: {tool_result.content[0].text}"
                    else:
                        bot_response = choice.get('content', "Sem resposta do modelo.")

                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    usage = data.get("usage", {})
                    tps = usage.get("completion_tokens", 0) / duration if duration > 0 else 0

                    return {
                        "status": "success", 
                        "response": bot_response,
                        "metrics": {
                            "total_tokens": usage.get("total_tokens", 0),
                            "tps": round(tps, 2),
                            "duration": round(duration, 2)
                        }
                    }

                except Exception as e:
                    return {"status": "error", "response": f"Erro na orquestração do Agente: {str(e)}"}