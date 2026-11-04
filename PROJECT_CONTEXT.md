# Projeto Meta 2 - Avaliação de LLMs

## Objetivo

Desenvolver um sistema de Question Answering para comparar dois modelos executados localmente via Ollama:

- AMALIA q4_K_M
- Llama 3.1 8B Instruct q4_K_M

## Hardware

- Intel i7-13700H
- 32 GB RAM
- RTX 4060 Laptop 8 GB

## Dataset

Formato:

tema,pergunta,metodo,resposta

onde:

- tema = categoria da pergunta
- pergunta = pergunta
- metodo = HUM ou LLM
- resposta = resposta correta

## Pipeline

Dataset
-> Inferência AMALIA
-> Inferência Llama
-> Recolha de respostas
-> Avaliação
-> Resultados
-> Gráficos

## Métricas

- Exact Match
- ROUGE-L
- BERTScore

## Ficheiros de saída

responses.csv

tema,metodo,pergunta,resposta_correta,resposta_amalia,resposta_llama

metrics.csv

tema,metodo,modelo,exact_match,rougeL,bertscore

## Requisitos

- Python 3.11+
- Código modular
- Type hints
- Docstrings
- PEP8

## Estrutura implementada

```text
llm-evaluation/
├── config.py
├── main.py
├── data/
├── results/
│   ├── responses/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── dataset/
│   ├── inference/
│   ├── evaluation/
│   ├── analysis/
│   ├── visualization/
│   └── utils/
├── scripts/
└── tests/
```

## Futuro

O projeto deverá ser facilmente extensível para Meta 3 (RAG).

## Implementação Meta 2

- Entrada: `data/dataset_PLN_test.tsv` por omissão; o carregador também aceita CSV e
  ignora colunas adicionais como `Autor`.
- Execução: `python main.py run`, ou os quatro scripts independentes em
  `scripts/`.
- Tags Ollama verificadas neste ambiente: `teex/amalia:q4_K_M` e
  `llama3.1:8b` (Q4_K_M).
- Saídas em `results/`: respostas, métricas por pergunta, médias globais/por
  tema/por método, três gráficos e metadados para reprodutibilidade.
- BERTScore usa `xlm-roberta-base` multilingue. Exact Match é uma comparação
  literal, sem normalização de maiúsculas, pontuação ou espaços.
- A geração implementa uma interface com contexto opcional para reutilização
  na futura Meta 3 (RAG).