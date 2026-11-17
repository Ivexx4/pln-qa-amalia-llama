# PLN 2026/2027 — Meta 2: resposta a perguntas

O projeto compara o AMALIA e o Llama 3.1 8B em perguntas de resposta curta, com avaliação lexical e semântica. Os conjuntos originais de treino e teste estão em `data/`; por omissão são avaliados em conjunto, sem treino dos modelos.

## Preparação

No ambiente virtual Python, instalar as dependências e os modelos no Ollama.

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull teex/amalia:q4_K_M
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

Linux/macOS (bash), criando o ambiente virtual se ainda não existir:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull teex/amalia:q4_K_M
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

Nos restantes comandos, `.\.venv\Scripts\python.exe` (Windows) equivale a `.venv/bin/python` (Linux/macOS). As versões das dependências diretas estão fixadas em `requirements.txt`; `scikit-learn` é necessário para o modo `--rag`.

O serviço Ollama tem de estar disponível em `http://localhost:11434`. O BERTScore usa o codificador [Albertina 100M PTPT](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptpt-encoder), adaptado com dados de português europeu, com a 12.ª camada. O modelo deriva do DeBERTa desenvolvido para inglês e reutiliza o seu tokenizador; não foi treinado exclusivamente de raiz em português. A escolha da camada não foi calibrada com avaliações humanas. O primeiro uso pode necessitar de acesso à rede para descarregar o modelo. Os pesos são convertidos para `float32` antes da avaliação; CUDA é utilizada quando disponível, caso contrário usa-se a CPU. Os modelos não devem ser incluídos na entrega.

## Execução e retoma

Windows (PowerShell):

```powershell
# Ensaio independente com uma pergunta por modelo
.\.venv\Scripts\python.exe main.py all --limit 1 --run-id verificacao-ptpt

# Experiência completa: 453 perguntas, dois modelos e respetivos julgamentos
.\.venv\Scripts\python.exe main.py all --run-id nova-ptpt
```

Linux/macOS (bash), com o ambiente virtual ativo:

```bash
# Ensaio independente com uma pergunta por modelo
python main.py all --limit 1 --run-id verificacao-ptpt

# Experiência completa: 453 perguntas, dois modelos e respetivos julgamentos
python main.py all --run-id nova-ptpt
```

As fases `generate`, `judge`, `score` e `analyze` podem ser executadas separadamente, com os mesmos argumentos e `--run-id`. Após interrupção, repetir o comando retoma as previsões e os vereditos já guardados. O manifesto impede a retoma se os dados, os modelos, os prompts, o código ou a configuração tiverem mudado; nesse caso, usar outro `--run-id`.

Para recalcular apenas as métricas de uma experiência compatível e completa, preservando as suas previsões e vereditos, criar uma análise separada. O exemplo pressupõe que as caches JSONL da execução `principal` estão disponíveis localmente; não são versionadas:

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe main.py score --run-id nova-estatistica --source-run principal
.\.venv\Scripts\python.exe main.py analyze --run-id nova-estatistica --source-run principal
```

Linux/macOS (bash):

```bash
python main.py score --run-id nova-estatistica --source-run principal
python main.py analyze --run-id nova-estatistica --source-run principal
```

`--source-run` verifica os dados, modelos, prompts e módulos usados para geração e julgamento, regista o hash do manifesto de origem e nunca altera a experiência original. Usar sempre esta opção nas fases `score` e `analyze` da análise derivada; se a origem não estiver completa, a pontuação falha, salvo com `--allow-partial`.

Para recalcular apenas as tabelas estatísticas de uma análise já exportada, sem repetir a geração, sem reescrever as restantes tabelas e sem revalidar o manifesto:

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe main.py analyze --run-id principal-ptpt --source-run principal --stats
```

Linux/macOS (bash):

```bash
python main.py analyze --run-id principal-ptpt --source-run principal --stats
```

O modo `--stats` lê o `all_results.csv` existente e regrava `tabela_estatistica.csv` e `tabela_por_tema_ic.csv`; não cria nem altera o manifesto, pelo que também pode ser usado em execuções anteriores à introdução de `stats.py`. `--source-run` é ignorado nesse modo: a proveniência das previsões está no manifesto da execução, que não é modificado.

Opções relevantes: `--split both|train|test`, `--limit`, `--train`, `--test`, `--models`, `--judge-model`, `--bert-model`, `--bert-num-layers`, `--bert-batch-size`, `--output-dir` e `--run-id`. `--allow-partial` permite avaliar uma execução incompleta, identificando-a como parcial; sem essa opção, previsões ou julgamentos em falta impedem a pontuação. `--stats` limita `analyze` às tabelas estatísticas. A comparação exigida na Meta 2 deve usar AMALIA e pelo menos outro LLM aberto com menos de 21 mil milhões de parâmetros.

### Validação humana do juiz

A fase `validate-judge` extrai uma amostra estratificada e reprodutível de 100 pares (pergunta, referência, resposta), com 50 vereditos CORRETO e 50 INCORRETO do juiz, usando seed 42. Só lê as caches de previsões e julgamentos: não escreve nelas, não cria manifesto e não altera análises. Se algum estrato tiver menos de 50 casos, usa o máximo disponível e avisa.

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe main.py validate-judge --run-id principal
```

Linux/macOS (bash):

```bash
python main.py validate-judge --run-id principal
```

O ficheiro `results/meta2/<run-id>/validacao_juiz.csv` tem as colunas `id`, `Modelo`, `Tema`, `Metodo`, `Pergunta`, `Gold`, `Pred`, `juiz` e `humano`; a última fica vazia para preencher com `1` (juiz correto) ou `0` (juiz incorreto). Depois de preenchida:

```powershell
.\.venv\Scripts\python.exe main.py judge-agreement --run-id principal
```

A fase `judge-agreement` exige `0` ou `1` em todas as linhas e grava `validacao_juiz_resumo.csv` com o número de casos, o acordo bruto, o kappa de Cohen, a precisão e o recall do juiz e a matriz de confusão (`tp`, `fp`, `fn`, `tn`), considerando CORRETO a classe positiva. Nenhuma destas fases entra em `all`.

### Auditoria das referências

`scripts/auditar_referencias.py` gera uma lista de casos a rever à mão, sem corrigir nem reescrever os TSV:

```powershell
.\.venv\Scripts\python.exe scripts\auditar_referencias.py
```

O resultado fica em `auditoria_referencias.csv` (`id`, `Tema`, `Pergunta`, `Resposta`, `motivo`, `gravidade`), ordenado por gravidade. Deteta perguntas duplicadas ou quase duplicadas (similaridade de Jaccard dos tokens ≥ 0,90), respostas de um único token muito curto, respostas sem tokens ou com caracteres inválidos, perguntas com anos de 2022 em diante ou expressões dependentes da data (`atual`, `atualmente`, `mais recente`), casos em que o juiz dá CORRETO com ROUGE-L abaixo de 0,1 e perguntas em que todos os modelos dão a mesma resposta sem conter a referência. A parte que depende dos resultados usa o `all_results.csv` mais recente das análises; `--results` permite escolher outro ficheiro, `--split` restringe a divisão e `--train`/`--test`/`--output` os restantes caminhos. O script só lê o dataset; as correções são sempre manuais.

Os dados de cada execução ficam em `results/meta2/<run-id>/`: o manifesto regista hashes dos ficheiros e modelos, versões das dependências, prompts e parâmetros; as caches JSONL registam a identidade das perguntas e respostas. As tabelas e gráficos ficam em `analises/<escopo>_completa/` ou `analises/<escopo>_parcial/`. Os resultados históricos diretamente em `results/` são preservados e não são reutilizados pela nova execução.

## Avaliação

- **EM**, **Contains** e **ROUGE-L F1** usam tokens normalizados sem distinção de maiúsculas, acentos e pontuação.
- **BERTScore F1** usa embeddings contextuais; **Judge** é um veredito automático do Qwen 2.5 7B em relação à resposta de referência.
- As tabelas agrupam por modelo, tema, método de origem da pergunta (`HUM`/`LLM`) e autor. As médias comparam apenas perguntas comuns a todos os modelos e, para cada métrica, apenas casos com valores disponíveis para todos; as colunas `n_*` indicam o número de valores usados.
- Respostas inválidas do juiz são identificadas e excluídas das médias, não consideradas incorretas. O juiz não substitui uma avaliação humana. As referências podem ser discutíveis ou sensíveis à data.
- `erros_juiz.csv` e `juiz_certo_metrica_lexical_baixa.csv` apoiam a análise de erros e discrepâncias.
- `tabela_estatistica.csv` contém médias e diferenças AMALIA − outro modelo para cada métrica, com IC percentil bootstrap a 95% (10 000 reamostragens, seed 42), emparelhado por pergunta nas diferenças. Apresenta testes bicaudais de McNemar exato (binomial) para Judge e Contains e Wilcoxon emparelhado para ROUGE-L e BERTScore. Os valores-p não são corrigidos para comparações múltiplas; não estabelecem correção factual.
- `tabela_por_tema_ic.csv` apresenta, por tema, modelo e métrica, a média, o IC, o número de perguntas comuns (`n_tema`) e o número com valores válidos para todos os modelos (`n`). Sinaliza `n<30` por métrica; com menos de duas observações, o IC fica vazio. Os IC de temas pequenos exigem especial cautela.

A inclusão de `stats.py` e das versões de NumPy e SciPy no manifesto altera a proveniência face às execuções anteriores. Para uma nova análise pelo CLI, escolher outro `--run-id` e usar `--source-run` com uma origem compatível; executar `score` e depois `analyze` para esse novo identificador.

## Estrutura e limites

`main.py` chama `qa_pipeline/cli.py`. A configuração central está em `qa_pipeline/config.py`. Os restantes módulos tratam dados, cliente Ollama, cache, geração, julgamento, métricas, recuperação lexical, análise e validação humana do juiz. Os testes são executados com `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` (Windows) ou `python -m unittest discover -s tests -v` (Linux/macOS, com o ambiente ativo).

O enunciado do projeto (`Projeto_Enunciado_26_27.pdf`) é fornecido pela unidade curricular e não é versionado; a cópia local está na pasta acima do repositório. Do diretório `results/` são versionados o `manifest.json` de cada experiência, o `all_results.csv` e as `tabela_*.csv` das análises; as caches `*.jsonl`, as figuras, os modelos e os ficheiros `validacao_juiz*.csv` (anotações humanas locais) ficam fora do repositório.

A opção `--rag` conserva uma experiência lexical anterior em `results/rag-experimental/`, separada da Meta 2. **Não cumpre a Meta 3**: esta exige documentos sobre pelo menos um tema, uma base de dados vetorial e avaliação da recuperação. O desenvolvimento dessa etapa fica para trabalho posterior.

O relatório apresenta os resultados completos de `principal-ptpt`, com BERTScore calculado usando Albertina a partir das previsões e dos vereditos de `principal`. Os resultados históricos diretamente em `results/` são distintos e não têm proveniência suficiente para confirmar o prompt do juiz.

## Meta 3: RAG com base de dados vetorial

Pipeline em `qa_pipeline/rag_*.py`, independente dos módulos da Meta 2 (que não foram alterados). Corpus: artigos da Wikipédia em português sobre um tema do dataset; embeddings `bge-m3` (Ollama); índice ChromaDB com distância do cosseno; os mesmos LLMs da Meta 2. O dataset **não** é indexado: o corpus só contém documentos externos.

```bash
ollama pull bge-m3
python -m pip install -r requirements.txt

# 1. Recolher o corpus (rede). Rever data/rag/<tema>/seeds.txt; o comando avisa dos títulos não encontrados.
python main.py rag corpus --theme "Futebol"

# 2. Ensaio com 2 perguntas e depois a experiência completa do tema (índice, recuperação, geração, juiz, métricas, comparação)
python main.py rag all --theme "Futebol" --limit 2 --run-id rag-ensaio
python main.py rag all --theme "Futebol" --run-id rag-v1
```

As fases podem ser corridas em separado (`index`, `retrieve`, `generate`, `judge`, `score`, `analyze`, `compare`) e são retomáveis. Parâmetros do RAG: ver a secção seguinte; `--baseline` (por omissão, o `all_results.csv` de `principal-ptpt`). Resultados em `results/meta3/<run-id>/`: `recuperacao_por_pergunta.csv` e `recuperacao_resumo.csv` (hit@k, MRR e cobertura de tokens da resposta no contexto), as tabelas habituais da Meta 2 e, na pasta de análise, `comparacao_rag_vs_baseline.csv` (diferença RAG − baseline por modelo e métrica, IC bootstrap, McNemar/Wilcoxon) e `efeito_rag_por_pergunta.csv` (onde o RAG ajudou ou prejudicou, cruzado com a recuperação).

### Configuração dos parâmetros

Todos os parâmetros do RAG estão num único objeto (`RagSettings`, em `qa_pipeline/rag_settings.py`), validado ao ser criado. A precedência, da menor para a maior, é: valores por omissão < ficheiro `--config` < opções da CLI < `--set chave=valor`.

| Parâmetro | Omissão | Afeta |
|---|---|---|
| `theme` | obrigatório | perguntas e corpus |
| `models`, `judge_model` | AMALIA + Llama 3.1 8B; Qwen 2.5 7B | geração, juízo |
| `embed_model` | `bge-m3:latest` | **índice** |
| `chunk_chars`, `chunk_overlap` | 900, 150 | **índice** |
| `top_k` | 4 | recuperação |
| `min_score` | nenhum | recuperação (descarta excertos com cosseno abaixo do limiar) |
| `store` | `chroma` | índice (`memory` = sem persistência) |
| `prompt` | `default` | geração (`default` ou `estrito`, em `rag_prompts.py`) |

Os parâmetros marcados como índice determinam a coleção ChromaDB (nome com hash do corpus e desses parâmetros), por isso variantes que só mudam `top_k`, `min_score` ou `prompt` reutilizam o mesmo índice sem recalcular embeddings. Sem `--run-id`, a execução chama-se `rag-<fingerprint>`, derivado dos parâmetros; o manifesto regista-os em `settings`.

```bash
python main.py rag all --config configs/rag/base.json --set top_k=6 --set min_score=0.4
python main.py rag sweep --sweep configs/rag/sweep_chunks_topk.json --limit 5
```

`sweep` expande a grelha do ficheiro JSON (`{"name", "base", "grid": {parâmetro: [valores]}}`) em variantes, corre `index` até `analyze` para cada uma e escreve `results/meta3/<nome>-resumo.csv` com recuperação e métricas lado a lado. Os valores da grelha têm precedência sobre a CLI. Para acrescentar um parâmetro novo: campo em `RagSettings` (com validação), uso no módulo respetivo e, se mudar o índice, inclusão em `INDEX_FIELDS`. Para um prompt novo: entrada em `PROMPTS`.

Limites a ter em conta: `hit@k` e `token_recall` são medidas lexicais (a resposta aparece no excerto), não provam que o excerto é relevante; com 20 perguntas por tema os IC são largos; o corpus e as sementes são escolhidos pelo autor, pelo que as sementes devem basear-se no tema e nas perguntas de treino, nunca nas de teste. A API do `chromadb` e as sementes iniciais não foram executadas contra o serviço real durante o desenvolvimento do código; validar com `--limit 2` antes da execução completa.

## Uso de ferramentas de assistência

O código, os testes e o relatório foram desenvolvidos com apoio do GitHub Copilot, no IDE, ao longo de muitos pedidos iterativos. A IA foi usada para a estrutura inicial do pipeline e dos módulos e como auxílio na validação dos resultados e dos erros; as decisões finais sobre o que alterar ou manter foram sempre tomadas pelo autor. O registo completo e ordenado dos pedidos não foi arquivado; por isso, não é possível determinar com exatidão todas as decisões individuais nem reproduzir a sequência integral de prompts. As sugestões foram executadas, comparadas com os dados e verificadas pela suite de testes; cada decisão apresentada é da responsabilidade do autor.