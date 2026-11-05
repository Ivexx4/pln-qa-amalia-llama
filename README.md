# PLN 2026/2027 – Meta 2 (Resposta a Perguntas)

## Preparação
```bash
pip install -r requirements.txt
ollama pull teex/amalia:q4_K_M
ollama pull llama3.1:8b
ollama pull qwen2.5:7b        # juiz (LLM-as-Judge)
```
Coloca `dataset_PLN_train.tsv` e `dataset_PLN_test.tsv` na pasta onde corres `main.py` (ou usa `--train`/`--test`).

## Execução
```bash
python main.py generate --limit 5   # teste rápido
python main.py all                  # pipeline completo (~453 perguntas x 2 LLMs)
```
Fases separadas: `generate`, `judge`, `score`, `analyze` (cache em `results/`, retoma se interrompido).
Opções: `--models`, `--judge-model`, `--bert-model`, `--split both|train|test`, `--train`, `--test`, `--limit`.

## Estrutura do código
```
main.py                  ponto de entrada
qa_pipeline/
  config.py              caminhos, modelos por omissão, parâmetros de decodificação
  prompts.py             prompts de geração e do juiz
  data.py                leitura e limpeza dos TSV
  llm.py                 cliente Ollama
  cache.py               cache JSONL (retoma execuções)
  generation.py          fase 1: respostas dos LLMs
  judge.py               fase 2: LLM-as-Judge
  metrics.py             EM, Contains, ROUGE-L
  scoring.py             junta tudo + BERTScore -> all_results.csv
  analysis.py            tabelas, gráficos e ficheiros de erros
  cli.py                 argumentos e orquestração das fases
```

## Decisões tomadas
- Modelos: AMALIA (q4_K_M) e Llama 3.1 8B (cabem em 8 GB de VRAM); juiz: Qwen 2.5 7B (modelo distinto dos avaliados).
- Dados: treino + teste juntos (453 perguntas), pois não há treino de modelos.
- Métricas: Exact Match (+ "Contains"), ROUGE-L (implementação própria, sem perder acentos), BERTScore (bert-base-multilingual-cased) e LLM-as-Judge.
- Decodificação determinística (temperature 0, seed 42), respostas curtas via prompt de sistema.
- Limpeza: artefactos `_x0096_`/`_x0080_` corrigidos; datas `YYYY-MM-DD 00:00:00` convertidas para "D de mês de AAAA".

## Saídas (`results/`)
`all_results.csv`, `tabela_por_{modelo,tema,metodo}.csv` (metodo = autor da pergunta, HUM vs LLM), gráficos `fig_*.png`,
`erros_juiz.csv`, `juiz_certo_metrica_lexical_baixa.csv`.

## Cuidados
- Algumas respostas de referência são discutíveis ou datadas (ex.: "país com maior taxa de sobrevivência ao cancro", campeões 2025/26); referir na análise de erros.
- Por tema, o número de perguntas é pequeno (cerca de 20 em média): interpretar com cautela.
- O LLM-as-Judge também erra: valida manualmente uma amostra (ex.: 30 casos) e reporta a concordância.
