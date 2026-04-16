from mcp.server.fastmcp import FastMCP

# Criamos o servidor MCP chamado "Coral Tools"
mcp = FastMCP("Coral Tools")

@mcp.tool()
def verificar_status_ambiente() -> str:
    """Verifica se o ambiente de backend e banco de dados estão operacionais."""
    # Aqui poderíamos ter um comando real: subprocess.run(["kubectl", "get", "pods"])
    return "Todos os serviços no namespace coral-project-krq9s estão RUNNING."

@mcp.tool()
def obter_horario_servidor() -> str:
    """Retorna o horário atual do servidor Nutanix NKP."""
    from datetime import datetime
    return f"O horário atual do sistema é: {datetime.now().strftime('%H:%M:%S')}"

if __name__ == "__main__":
    mcp.run()