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
- **Endpoints salvos e recentes**: um seletor preenche URL, modelo e chave. Os salvos (⭐) vêm de variáveis
  de ambiente do backend; os recentes (🕘) são os usados com sucesso na sessão. O app gera o YAML para salvar
  os recentes no Kubernetes.
- **Modo comparação**: a mesma pergunta para dois endpoints ou modelos lado a lado, com resumo de quem foi mais rápido.
  O envio pode ser simultâneo ou sequencial (B só começa quando A termina, para um não disputar recursos com o outro).
- **Modo benchmark**: repete a mesma pergunta várias vezes e mede o endpoint com estatística (veja abaixo).
- **Documento anexado** (PDF, TXT, MD, CSV, JSON, YAML, LOG): o texto vai como contexto em cada pergunta. PDFs escaneados não são suportados.
- **Exportar a conversa** em Markdown.

Ferramentas (MCP) não ficam mais no app: use as conexões MCP configuradas no próprio NAI.

## Endpoints salvos

O app não grava variáveis de ambiente em tempo de execução: os endpoints salvos são definidos no
`deployment.yaml` (backend) e carregados quando o pod sobe. Cada perfil usa um número:

```yaml
- name: NAI_PROFILE_1_NAME
  value: "GPU L40S - Llama 3.1 8B"
- name: NAI_PROFILE_1_URL
  value: "https://<nai>/enterpriseai/gateway/v1/chat/completions"
- name: NAI_PROFILE_1_MODEL
  value: "llama-31-8b"
- name: NAI_PROFILE_1_API_KEY
  valueFrom:
    secretKeyRef:
      name: nai-endpoints
      key: profile-1-api-key
      optional: true
```

```bash
kubectl create secret generic nai-endpoints -n <namespace> \
  --from-literal=profile-1-api-key='<API_KEY>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

O endpoint padrão (`NAI_ENDPOINT`, `MODEL_NAME`, `NAI_API_KEY`) também aparece no seletor. Na barra lateral,
**🗂️ Salvar endpoints no Kubernetes** gera esses trechos a partir dos endpoints usados na sessão.

Segurança das chaves: as API Keys dos perfis ficam só no backend (a rota `/profiles` não as devolve e elas
não são preenchidas na tela). Uma chave salva só é enviada para o mesmo host (`scheme://host:porta`) do
perfil, então trocar a URL para outro endereço não leva a chave junto.

## Benchmark

O modo **📈 Benchmark** mede um endpoint (ou compara dois) repetindo o mesmo prompt. Há três tipos de teste:

- **Rodada única**: uma configuração (requisições, concorrência, aquecimento, máximo de tokens, temperatura,
  prefixo anti-cache e tamanho fixo de saída com `ignore_eos`, do vLLM).
- **Varredura de concorrência**: repete a medição com vários níveis (ex.: 1, 2, 4, 8, 16 usuários
  simultâneos), com (fator × concorrência) requisições por nível.
- **Varredura de contexto**: repete com prompts de vários tamanhos (ex.: 256, 1k, 4k, 16k tokens). O tamanho
  é calibrado com o tokenizador do próprio modelo e o valor real (do `usage`) é o que vai para os gráficos.

Métricas por requisição: tempo até o 1º token (TTFT), tempo total, tempo por token de saída (TPOT),
velocidade de geração, intervalo entre chunks e tokens de entrada/saída. Estatísticas: média, desvio, CV,
mínimo, p50, p90, p99, máximo, throughput agregado e **throughput em regime** (só o trecho com todas as vagas
de concorrência ocupadas, sem a subida e a descida da carga).

Gráficos:

- Rodada única: linha do tempo das requisições (fila + prefill × geração), requisições em andamento ao longo
  do tempo, tokens/s ao longo do tempo, KV cache e fila do servidor, distribuições (histogramas) e valores
  por requisição.
- Varredura de concorrência: throughput, TTFT p50/p90, geração por usuário e tempo total por nível;
  curva latência × throughput; pico de KV cache e de fila por nível.
- Varredura de contexto: TTFT, velocidade de prefill, geração por usuário e tempo total por tamanho de
  contexto; pico de KV cache por tamanho.

Tudo pode ser exportado em Markdown (relatório) e CSV (dados brutos).

### KV cache, fila e cache de prefixo

Essas métricas não aparecem na API de chat: ficam no endpoint `/metrics` do servidor de inferência (vLLM).
Informe essa URL na barra lateral (campo **URL de métricas do servidor**, visível no modo Benchmark) e use
**📈 Testar métricas** para conferir. Durante o benchmark, o backend lê o `/metrics` a cada 0,5 s e mostra:
uso do KV cache (`vllm:kv_cache_usage_perc`), requisições em execução e na fila, taxa de acerto do cache de
prefixo e preempções. Nomes antigos (`gpu_cache_usage_perc`) e sem o prefixo `vllm:` também são aceitos.

A URL precisa ser acessível a partir do pod do backend. Se o NAI roda no mesmo cluster, costuma ser o
Service interno do pod de inferência (algo como `http://<servico>.<namespace>.svc:<porta>/metrics`). A URL
pode ficar salva no endpoint: `NAI_PROFILE_<N>_METRICS_URL` (ou `NAI_METRICS_URL` para o padrão).

### Precisão

O benchmark roda no backend, dentro do cluster, e não no navegador. O relógio é monotônico, o TTFT começa a
contar quando a requisição termina de ser enviada, as conexões são reaproveitadas, o aquecimento é
descartado e os tokens vêm do campo `usage` do NAI. Validado contra um servidor com tempos conhecidos: o
tempo por token bate com a referência (erro abaixo de 0,1%), o TTFT fica a cerca de 1 ms do medido por um
cliente de socket cru, a soma da linha do tempo bate com o total de tokens e os tokens contados pelo
cliente são iguais aos contadores do servidor (`/metrics`).

## Variáveis de ambiente

Todas são opcionais; sem elas o app usa os padrões abaixo.

| Variável | Onde | Padrão | Uso |
|---|---|---|---|
| `NAI_ENDPOINT` | backend e frontend | gateway de demo | URL de chat completions do NAI |
| `NAI_API_KEY` | backend e frontend | vazio | API Key do endpoint padrão (usada só no mesmo host) |
| `NAI_PROFILE_<N>_NAME`, `_URL`, `_MODEL`, `_API_KEY`, `_METRICS_URL` | backend | — | Endpoints salvos (veja acima) |
| `NAI_METRICS_URL` | backend | vazio | `/metrics` do servidor do endpoint padrão (KV cache no benchmark) |
| `NAI_PROFILE_DEFAULT_NAME` | backend | `Padrão (manifest)` | Nome do endpoint padrão no seletor |
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
| `POST /api/v1/chat/benchmark` | Benchmark do endpoint (NDJSON: `start`, `calibration`, `warmup`, `result`, `summary`, `error`, `done`) |
| `POST /api/v1/chat/metrics-check` | Lê uma vez o `/metrics` do servidor de inferência e lista o que encontrou |
| `GET /api/v1/chat/profiles` | Endpoints salvos (sem as API Keys) |
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
