"""Prompt templates for Portuguese-European QA."""

SYSTEM_PROMPT = (
    "Responde em português europeu, de forma direta e concisa. "
    "Dá apenas a resposta à pergunta; não inventes informação."
)


def build_user_prompt(question: str, context: str | None = None) -> str:
    """Build the user message, optionally grounding it in retrieved context."""
    if context:
        return f"Contexto:\n{context}\n\nPergunta:\n{question}"
    return question


def build_system_prompt(context: str | None = None) -> str:
    """Return the system message for standard QA or context-grounded QA."""
    if context:
        return (
            SYSTEM_PROMPT
            + " Usa o contexto fornecido como fonte principal e indica "
            "quando não contém informação suficiente."
        )
    return SYSTEM_PROMPT
