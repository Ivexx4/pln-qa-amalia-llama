"""Prompts usados na geração de respostas e no LLM-as-Judge."""

SYSTEM_PROMPT = (
    "És um assistente que responde a perguntas de cultura geral em português europeu. "
    "Responde apenas com uma resposta curta (no máximo uma frase), sem explicações adicionais."
)

RAG_SYSTEM_PROMPT = (
    "És um assistente que responde a perguntas de cultura geral em português europeu. "
    "Usa o contexto relevante fornecido para fundamentar a resposta, mas mantém a resposta curta "
    "(no máximo uma frase) e sem explicações extras."
)

JUDGE_PROMPT = """Avalia se a resposta candidata está correta face à resposta de referência.

Pergunta: {q}
Resposta de referência: {gold}
Resposta candidata: {pred}

Considera a resposta candidata correta se transmite a mesma informação essencial que a referência
(ignora diferenças de formulação, nível de detalhe ou formato de datas). Se contradiz a referência,
omite a informação pedida ou não responde, é incorreta.
Responde apenas com um objeto JSON válido, sem texto adicional, neste formato:
{{"veredito": "CORRETO"}}
O valor de "veredito" tem de ser exatamente "CORRETO" ou "INCORRETO"."""
