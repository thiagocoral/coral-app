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

    # 1. LOGICA DE FILTRO PREVENTIVO
    user_query = message.user_input.lower()
    tech_keywords = ["hora", "horário", "recursos", "memória", "cpu", "arquivos", "listar", "conectividade", "ping", "status"]
    
    # Se NÃO houver palavras técnicas, forçamos o modelo a não usar ferramentas (ignora alucinação no "olá")
    use_tools_allowed = any(k in user_query for k in tech_keywords)
    final_tool_choice = "auto" if use_tools_allowed else "none"

    # Inicia a sessão com o Servidor MCP via STDIO
    async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Busca ferramentas disponíveis
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

            headers = {
                "Authorization": f"Bearer {final_key}", 
                "Content-Type": "application/json"
            }
            
            # Monta o payload respeitando a decisão do filtro
            payload = {
                "model": final_model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "Você é o assistente NTNX BR. Responda saudações educadamente apenas com texto. Use ferramentas apenas para solicitações técnicas de sistema."
                    },
                    *message.history, 
                    {"role": "user", "content": message.user_input}
                ],
                "temperature": 0.1,  # Baixa temperatura para maior precisão
                "tools": nai_tools,
                "tool_choice": final_tool_choice
            }

            async with httpx.AsyncClient(verify=False) as client:
                try:
                    response = await client.post(final_endpoint, json=payload, headers=headers, timeout=120.0)
                    
                    if response.status_code != 200:
                        return {"status": "error", "response": f"Erro NAI: {response.text}"}

                    data = response.json()
                    choice = data['choices'][0]['message']

                    # 2. TRATAMENTO DA RESPOSTA
                    # Se houver tool_calls E estivermos em modo permitido
                    if choice.get("tool_calls") and use_tools_allowed:
                        tool_call = choice["tool_calls"][0]
                        tool_name = tool_call["function"]["name"]
                        
                        # Parse dos argumentos
                        args_raw = tool_call["function"]["arguments"]
                        tool_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        
                        # Executa a ferramenta
                        tool_result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        # Extrai o texto do resultado
                        if hasattr(tool_result, 'content') and len(tool_result.content) > 0:
                            resultado_texto = tool_result.content[0].text
                        else:
                            resultado_texto = str(tool_result)

                        bot_response = f"🛠️ **Ação do Agente:** {tool_name}\n\n✅ **Resultado:** {resultado_texto}"
                    
                    else:
                        # Se não for ferramenta, retorna o conteúdo de texto normal
                        bot_response = choice.get('content', "")
                        
                        # Fallback: se a IA retornar vazio mas tentar mandar um JSON de tool_call indevido
                        if not bot_response and choice.get("tool_calls"):
                            bot_response = "Olá! Como posso ajudar você com informações técnicas do sistema hoje?"

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
                    return {"status": "error", "response": f"Erro na orquestração: {str(e)}"}