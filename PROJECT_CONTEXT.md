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

## Estrutura pretendida

src/

dataset/
inference/
evaluation/
visualization/
utils/

## Futuro

O projeto deverá ser facilmente extensível para Meta 3 (RAG).