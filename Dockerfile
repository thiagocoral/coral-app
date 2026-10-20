FROM python:3.10-slim

# Impede que o Python gere arquivos .pyc e garante que os logs/stdio saiam em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Adiciona o diretório /app ao PYTHONPATH para que os módulos se encontrem
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]
