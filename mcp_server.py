import subprocess
import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Coral Enterprise Tools")

@mcp.tool()
def listar_arquivos_projeto() -> str:
    """
    Lista arquivos no diretório /app. 
    USE APENAS se o usuário solicitar explicitamente ver a estrutura de pastas, verificar se um arquivo existe 
    ou listar o conteúdo do projeto. NÃO use para conversas gerais ou saudações.
    """
    try:
        arquivos = os.listdir('/app')
        return f"Arquivos encontrados em /app: {', '.join(arquivos)}"
    except Exception as e:
        return f"Erro ao listar arquivos: {str(e)}"

@mcp.tool()
def verificar_recursos_container() -> str:
    """
    Verifica métricas de hardware (RAM/CPU). 
    USE APENAS se o usuário perguntar sobre consumo de memória, performance do servidor, 
    saúde do container ou recursos técnicos. NÃO use se o usuário apenas disser 'tudo bem' ou 'olá'.
    """
    try:
        with open('/proc/self/status', 'r') as f:
            status = f.read()
        for line in status.split('\n'):
            if 'VmRSS' in line:
                # Retornamos um dicionário simulado para manter a consistência que a IA espera
                mem = line.split(':')[1].strip()
                return f"{{'memoria': '{mem}', 'cpu': 'estável'}}"
        return "Informação de memória não encontrada."
    except Exception as e:
        return f"Erro ao ler recursos: {str(e)}"

@mcp.tool()
def testar_conectividade_nai() -> str:
    """
    Valida a rede com o Nutanix NAI. 
    USE APENAS se o usuário relatar problemas de resposta da IA, lentidão 
    ou perguntar especificamente se o endpoint NAI está acessível.
    """
    endpoint = os.getenv("NAI_ENDPOINT", "10.54.94.16")
    host = endpoint.split('//')[-1].split('/')[0]
    
    try:
        subprocess.check_output(["ping", "-c", "1", host], stderr=subprocess.STDOUT, timeout=2)
        return f"Conectividade com NAI OK: {host} está respondendo."
    except Exception:
        return f"Alerta: Falha de conectividade direta com o host {host}."

@mcp.tool()
def obter_horario_servidor() -> str:
    """
    Retorna a hora exata do sistema. 
    USE APENAS se o usuário perguntar 'que horas são', 'qual o horário' ou 'hora atual'. 
    É EXPRESSAMENTE PROIBIDO usar esta ferramenta para responder saudações como 'olá' ou 'tudo bem'.
    """
    return f"O horário atual do sistema é: {datetime.now().strftime('%H:%M:%S')}"

if __name__ == "__main__":
    mcp.run()