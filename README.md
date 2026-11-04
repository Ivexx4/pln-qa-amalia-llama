# LLM Evaluation — Meta 2

Pipeline de Question Answering para comparar AMALIA e Llama 3.1 8B Instruct,
executados localmente através do Ollama. A estrutura separa carregamento dos
dados, inferência, avaliação, análise e visualização; a interface do cliente
aceita contexto opcional para ser reutilizada na Meta 3 (RAG).

## Estrutura

```text
llm-evaluation/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── data/
│   ├── dataset_PLN_train.tsv
│   └── dataset_PLN_test.tsv
├── results/
│   ├── responses/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── dataset/       # schema e carregamento CSV/TSV
│   ├── inference/     # prompts, Ollama e geração
│   ├── evaluation/    # Exact Match, ROUGE-L e BERTScore
│   ├── analysis/      # agregações globais, por tema e por método
│   ├── visualization/ # gráficos Matplotlib
│   └── utils/         # logging e I/O
├── scripts/           # comandos para cada etapa
└── tests/
```

## Instalação

Requer Python 3.11+, Ollama e os modelos disponíveis localmente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama list
```

Os nomes de modelo predefinidos (`teex/amalia:q4_K_M` e `llama3.1:8b`) foram
verificados neste projeto. Se as tags diferirem na instalação local, passa
outras tags em `--amalia-model` e `--llama-model`.

## Executar

O comando completo usa o conjunto de teste por omissão:

```powershell
.\.venv\Scripts\python.exe main.py run
```

Para validar a instalação com três perguntas antes da execução completa:

```powershell
.\.venv\Scripts\python.exe main.py run --limit 3 --device cpu
```

Cada etapa também pode ser executada em separado:

```powershell
.\.venv\Scripts\python.exe scripts\01_generate_responses.py
.\.venv\Scripts\python.exe scripts\02_evaluate_models.py --device auto
.\.venv\Scripts\python.exe scripts\03_generate_analysis.py
.\.venv\Scripts\python.exe scripts\04_generate_plots.py
```

O diretório de entrada por omissão é `data/`. São suportados CSV e TSV, os
cabeçalhos portugueses da Meta 1 e colunas adicionais como `Autor`.
BERTScore utiliza o modelo multilingue `xlm-roberta-base`, adequado para
português. A primeira avaliação descarrega os pesos do Hugging Face. `--device
auto` usa CUDA quando disponível; use `--device cpu` ou um batch menor se faltar
memória na GPU. A versão de PyTorch instalada deve suportar CUDA para aproveitar
a RTX 4060.

## Resultados

Os ficheiros são colocados em:

- `results/responses/responses.csv`: pergunta, ground truth e respostas dos dois
  modelos; atualizado à medida que a inferência avança.
- `results/metrics/metrics.csv`: métricas por pergunta e modelo.
- `results/metrics/summary_global.csv`, `summary_tema.csv` e
  `summary_metodo.csv`: médias e contagem de exemplos por grupo.
- `results/plots/`: gráficos global, por tema e HUM vs LLM.
- `results/responses/run_metadata.json`: hash do dataset, tags e parâmetros de
  geração para ajudar a reproduzir a experiência.

Métricas: Exact Match literal, ROUGE-L F-measure sem stemming e BERTScore F1
sem reescalamento por baseline. Exact Match é sensível a diferenças de
maiúsculas, pontuação e espaços; interpreta-a em conjunto com as métricas
semânticas.

Para a Meta 3, implemente `AnswerGenerator.generate(question, context=None)` em
`src/inference/ollama_client.py` ou acrescente um cliente RAG que satisfaça a
mesma interface. O carregamento, a avaliação e a produção de relatórios podem
ser reutilizados.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
