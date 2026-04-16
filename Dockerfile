FROM python:3.9-slim

# Impede que o Python gere arquivos .pyc e garante que os logs/stdio saiam em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Adiciona o diretório /app ao PYTHONPATH para que os módulos se encontrem
ENV PYTHONPATH=/app

WORKDIR /app

# Instala dependências do sistema caso o MCP precise de pacotes extras (opcional, mas recomendado)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Substitua a linha do pip por esta:
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copia todo o projeto (incluindo o mcp_server.py na raiz)
COPY . .

EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]