# coral-app

Chat para endpoints do **Nutanix Enterprise AI (NAI)**, implantado no **NKP**.

- `core/` — backend FastAPI (porta 8000, serviço `app-service`). Fala com o endpoint NAI (API compatível com OpenAI).
- `frontend/` — interface Streamlit (porta 8501, serviço `frontend-service`). Fala só com o backend.

O pipeline em `.github/workflows/docker-build.yml` publica `thicoral/coral-app:v1` (backend) e
`thicoral/coral-app:frontend` (frontend) a cada push na `main`. Os manifests `deployment.yaml` e
`frontend-deployment.yaml` são os usados no deploy pelo NKP.

## Recursos do chat

- **Streaming real** das respostas, token a token.
- **Métricas por resposta**: tempo até o primeiro token (TTFT), tokens/s de geração, tokens de prompt e de resposta, tempo total.
  Quando o endpoint não informa o uso de tokens, os valores aparecem com `~` (estimados).
- **Memória da conversa**: o histórico vai junto em cada pergunta (limitado por `MAX_HISTORY_MESSAGES`).
- **Raciocínio do modelo** em seção recolhível, para modelos que o retornam (`reasoning_content`, `reasoning` ou tags `<think>`).
- **Parâmetros**: system prompt, temperatura e máximo de tokens, ajustáveis na barra lateral.
- **Lista de modelos** do endpoint (`/v1/models`) e **teste de conexão** (rede, TLS e API Key).
- **Modo comparação**: a mesma pergunta para dois endpoints ou modelos lado a lado, com resumo de quem foi mais rápido.
  O envio pode ser simultâneo ou sequencial (B só começa quando A termina, para um não disputar recursos com o outro).
- **Modo benchmark**: repete a mesma pergunta várias vezes e mede o endpoint com estatística (veja abaixo).
- **Documento anexado** (PDF, TXT, MD, CSV, JSON, YAML, LOG): o texto vai como contexto em cada pergunta. PDFs escaneados não são suportados.
- **Exportar a conversa** em Markdown.

Ferramentas (MCP) não ficam mais no app: use as conexões MCP configuradas no próprio NAI.

## Benchmark

O modo **📈 Benchmark** mede um endpoint (ou compara dois) repetindo o mesmo prompt:

- Configurável: número de requisições, concorrência, aquecimento, máximo de tokens, temperatura,
  prefixo anti-cache e tamanho fixo de saída (`ignore_eos`, do vLLM).
- Métricas por requisição: tempo até o 1º token (TTFT), tempo total, tempo por token de saída (TPOT),
  velocidade de geração (tok/s), intervalo entre chunks e tokens de entrada/saída.
- Estatísticas: média, desvio padrão, coeficiente de variação, mínimo, p50, p90, p99 e máximo, além do
  throughput agregado (tokens/s somando todas as requisições).
- Exporta relatório em Markdown e os dados brutos em CSV.

Para a medição ser precisa, o benchmark roda no backend, dentro do cluster, e não no navegador. O relógio
é monotônico, o TTFT começa a contar quando a requisição termina de ser enviada, as conexões são
reaproveitadas, o aquecimento é descartado e os tokens vêm do campo `usage` do NAI. Validado contra um
servidor com tempos conhecidos: o tempo por token bate com a referência (erro abaixo de 0,1%) e o TTFT
fica a cerca de 1 ms do medido por um cliente de socket cru.

## Variáveis de ambiente

Todas são opcionais; sem elas o app usa os padrões abaixo.

| Variável | Onde | Padrão | Uso |
|---|---|---|---|
| `NAI_ENDPOINT` | backend e frontend | gateway de demo | URL de chat completions do NAI |
| `NAI_API_KEY` | backend e frontend | vazio | API Key do endpoint |
| `MODEL_NAME` | backend e frontend | `unified-endpoint` | Modelo padrão |
| `NAI_ENDPOINT_B`, `NAI_API_KEY_B`, `MODEL_NAME_B` | frontend | iguais ao endpoint A | Valores iniciais do endpoint B no modo comparação |
| `BACKEND_URL` | frontend | `http://app-service` | Endereço do backend |
| `SYSTEM_PROMPT` | backend | assistente NTNX BR | System prompt padrão |
| `DEFAULT_TEMPERATURE` | backend | `0.3` | Temperatura padrão |
| `DEFAULT_MAX_TOKENS` | backend | `2048` | Máximo de tokens padrão |
| `MAX_HISTORY_MESSAGES` | backend | `20` | Mensagens de histórico enviadas ao modelo |
| `DOC_MAX_CHARS` | backend | `20000` | Limite de caracteres do documento anexado |
| `NAI_VERIFY_SSL` | backend | `false` | `true` para validar o certificado do NAI |
| `NAI_CA_BUNDLE` | backend | vazio | Caminho de um CA para validar o certificado |
| `NAI_REQUEST_TIMEOUT` | backend | `300` | Timeout de leitura (segundos) |

Para não deixar a API Key no manifest, ela pode vir de um Secret:

```yaml
- name: NAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: nai-credentials
      key: api-key
      optional: true
```

## API do backend

| Rota | Descrição |
|---|---|
| `GET /health` | Health check (probes do Kubernetes) |
| `POST /api/v1/chat/stream` | Resposta em streaming (NDJSON: `reasoning`, `content`, `reclassify`, `metrics`, `notice`, `error`, `done`) |
| `POST /api/v1/chat/ask` | Resposta completa, sem streaming |
| `POST /api/v1/chat/models` | Modelos do endpoint |
| `POST /api/v1/chat/connectivity` | Teste de rede, TLS e autenticação |
| `POST /api/v1/chat/benchmark` | Benchmark do endpoint (NDJSON: `start`, `warmup`, `result`, `summary`, `error`, `done`) |
| `GET /api/v1/chat/defaults` | Padrões configurados (sem a API Key) |

## Rodar localmente

```bash
pip install -r requirements.txt
export NAI_ENDPOINT="https://<nai>/enterpriseai/gateway/v1/chat/completions" NAI_API_KEY="<chave>"
PYTHONPATH=. uvicorn core.main:app --port 8000
BACKEND_URL=http://localhost:8000 streamlit run frontend/frontend.py
```

## Testes

```bash
pip install -r requirements-dev.txt
python -m pytest tests
```

Os testes usam um NAI simulado (`tests/fake_nai.py`) e não precisam de rede.

## Comandos úteis

```bash
export KUBECONFIG="./workload01-kubeconfig (2).conf"

kubectl delete pods -n coral-project-8lxdv --all
kubectl rollout restart deployment coral-app -n coral-project-8lxdv
kubectl rollout restart deployment coral-frontend -n coral-project-8lxdv
kubectl get pods -n coral-project-8lxdv
```

Como as tags das imagens são fixas (`v1` e `frontend`), depois de um push na `main` é preciso reiniciar os
deployments para puxar a imagem nova.

Last 100% commit: 386c729cd3a10993c0f250dcf8537e4b465da066
