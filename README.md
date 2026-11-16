# PLN 2026/2027 — Meta 2: resposta a perguntas

O projeto compara o AMALIA e o Llama 3.1 8B em perguntas de resposta curta, com avaliação lexical e semântica. Os conjuntos originais de treino e teste estão em `data/`; por omissão são avaliados em conjunto, sem treino dos modelos.

## Preparação

No ambiente virtual Python, instalar as dependências e os modelos no Ollama:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull teex/amalia:q4_K_M
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

O serviço Ollama tem de estar disponível em `http://localhost:11434`. O BERTScore usa o codificador [Albertina 100M PTPT](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptpt-encoder), adaptado com dados de português europeu, com a 12.ª camada. O modelo deriva do DeBERTa desenvolvido para inglês e reutiliza o seu tokenizador; não foi treinado exclusivamente de raiz em português. A escolha da camada não foi calibrada com avaliações humanas. O primeiro uso pode necessitar de acesso à rede para descarregar o modelo. Os pesos são convertidos para `float32` antes da avaliação; CUDA é utilizada quando disponível, caso contrário usa-se a CPU. Os modelos não devem ser incluídos na entrega.

## Execução e retoma

```powershell
# Ensaio independente com uma pergunta por modelo
.\.venv\Scripts\python.exe main.py all --limit 1 --run-id verificacao-ptpt

# Experiência completa: 453 perguntas, dois modelos e respetivos julgamentos
.\.venv\Scripts\python.exe main.py all --run-id nova-ptpt
```

As fases `generate`, `judge`, `score` e `analyze` podem ser executadas separadamente, com os mesmos argumentos e `--run-id`. Após interrupção, repetir o comando retoma as previsões e os vereditos já guardados. O manifesto impede a retoma se os dados, os modelos, os prompts, o código ou a configuração tiverem mudado; nesse caso, usar outro `--run-id`.

Para recalcular apenas as métricas de uma experiência compatível e completa, preservando as suas previsões e vereditos, criar uma análise separada:

```powershell
.\.venv\Scripts\python.exe main.py score --run-id principal-ptpt --source-run principal
.\.venv\Scripts\python.exe main.py analyze --run-id principal-ptpt --source-run principal
```

`--source-run` verifica os dados, modelos, prompts e módulos usados para geração e julgamento, regista o hash do manifesto de origem e nunca altera a experiência original. Usar sempre esta opção nas fases `score` e `analyze` da análise derivada; se a origem não estiver completa, a pontuação falha, salvo com `--allow-partial`.

Opções relevantes: `--split both|train|test`, `--limit`, `--train`, `--test`, `--models`, `--judge-model`, `--bert-model`, `--bert-num-layers`, `--bert-batch-size`, `--output-dir` e `--run-id`. `--allow-partial` permite avaliar uma execução incompleta, identificando-a como parcial; sem essa opção, previsões ou julgamentos em falta impedem a pontuação. A comparação exigida na Meta 2 deve usar AMALIA e pelo menos outro LLM aberto com menos de 21 mil milhões de parâmetros.

Os dados de cada execução ficam em `results/meta2/<run-id>/`: o manifesto regista hashes dos ficheiros e modelos, versões das dependências, prompts e parâmetros; as caches JSONL registam a identidade das perguntas e respostas. As tabelas e gráficos ficam em `analises/<escopo>_completa/` ou `analises/<escopo>_parcial/`. Os resultados históricos diretamente em `results/` são preservados e não são reutilizados pela nova execução.

## Avaliação

- **EM**, **Contains** e **ROUGE-L F1** usam tokens normalizados sem distinção de maiúsculas, acentos e pontuação.
- **BERTScore F1** usa embeddings contextuais; **Judge** é um veredito automático do Qwen 2.5 7B em relação à resposta de referência.
- As tabelas agrupam por modelo, tema, método de origem da pergunta (`HUM`/`LLM`) e autor. As médias comparam apenas perguntas comuns a todos os modelos e, para cada métrica, apenas casos com valores disponíveis para todos; as colunas `n_*` indicam o número de valores usados.
- Respostas inválidas do juiz são identificadas e excluídas das médias, não consideradas incorretas. O juiz não substitui uma avaliação humana. As referências podem ser discutíveis ou sensíveis à data.
- `erros_juiz.csv` e `juiz_certo_metrica_lexical_baixa.csv` apoiam a análise de erros e discrepâncias.

## Estrutura e limites

`main.py` chama `qa_pipeline/cli.py`. A configuração central está em `qa_pipeline/config.py`; `config.py` na raiz mantém compatibilidade. Os restantes módulos tratam dados, cliente Ollama, cache, geração, julgamento, métricas e análise. Os testes são executados com `python -m unittest discover -s tests -v`.

A opção `--rag` conserva uma experiência lexical anterior em `results/rag-experimental/`, separada da Meta 2. **Não cumpre a Meta 3**: esta exige documentos sobre pelo menos um tema, uma base de dados vetorial e avaliação da recuperação. O desenvolvimento dessa etapa fica para trabalho posterior.

O relatório apresenta os resultados completos de `principal-ptpt`, com BERTScore calculado usando Albertina a partir das previsões e dos vereditos de `principal`. Os resultados históricos diretamente em `results/` são distintos e não têm proveniência suficiente para confirmar o prompt do juiz.

## Uso de ferramentas de assistência

O código, os testes e o relatório foram desenvolvidos com apoio do GitHub Copilot, no IDE, ao longo de muitos pedidos iterativos. A IA foi usada para a estrutura inicial do pipeline e dos módulos e como auxílio na validação dos resultados e dos erros; as decisões finais sobre o que alterar ou manter foram sempre tomadas pelo autor. O registo completo e ordenado dos pedidos não foi arquivado; por isso, não é possível determinar com exatidão todas as decisões individuais nem reproduzir a sequência integral de prompts. As sugestões foram executadas, comparadas com os dados e verificadas pela suite de testes; cada decisão apresentada é da responsabilidade do autor.