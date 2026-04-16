import subprocess
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Coral Enterprise Tools")

@mcp.tool()
def listar_arquivos_projeto() -> str:
    """Lista todos os arquivos na raiz do projeto dentro do container."""
    try:
        arquivos = os.listdir('/app')
        return f"Arquivos encontrados em /app: {', '.join(arquivos)}"
    except Exception as e:
        return f"Erro ao listar arquivos: {str(e)}"

@mcp.tool()
def verificar_recursos_container() -> str:
    """Verifica o uso de memória e CPU do container atual (simulado via proc)."""
    try:
        # Lê o uso de memória do container
        with open('/proc/self/status', 'r') as f:
            status = f.read()
        # Filtra apenas a linha de memória virtual usada (VmRSS)
        for line in status.split('\n'):
            if 'VmRSS' in line:
                return f"Uso de Memória Atual: {line.split(':')[1].strip()}"
        return "Informação de memória não encontrada."
    except Exception as e:
        return f"Erro ao ler recursos: {str(e)}"

@mcp.tool()
def testar_conectividade_nai() -> str:
    """Realiza um teste de ping/conexão simples para o endpoint do Nutanix NAI."""
    # O endpoint está na variável de ambiente NAI_ENDPOINT
    endpoint = os.getenv("NAI_ENDPOINT", "10.54.94.16")
    # Limpa a URL para pegar apenas o IP/Host
    host = endpoint.split('//')[-1].split('/')[0]
    
    try:
        # Executa um ping simples (1 pacote)
        output = subprocess.check_output(["ping", "-c", "1", host], stderr=subprocess.STDOUT, timeout=2)
        return f"Conectividade com NAI OK: {host} está respondendo."
    except Exception:
        return f"Alerta: Falha de conectividade direta com o host {host}."

@mcp.tool()
def obter_horario_servidor() -> str:
    """
    RETORNA O HORÁRIO ATUAL. 
    USE ESTA FERRAMENTA APENAS SE o usuário perguntar explicitamente 'que horas são', 
    'qual o horário' ou 'hora atual'. NÃO use para saudações ou reclamações.
    """
    from datetime import datetime
    return f"O horário atual do sistema é: {datetime.now().strftime('%H:%M:%S')}"


if __name__ == "__main__":
    mcp.run()
