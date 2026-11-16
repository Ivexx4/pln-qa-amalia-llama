"""Métricas lexicais com normalização de maiúsculas, acentos e pontuação."""
import re
import unicodedata


def tokens(s: str) -> list:
    """Minúsculas, sem acentos nem pontuação, preservando letras e dígitos Unicode."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", s).split()


def lcs(a: list, b: list) -> int:
    """Comprimento da maior subsequência comum."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    n = lcs(p, g)
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> int:
    """1 apenas se as sequências de tokens coincidirem e a referência tiver tokens."""
    g = tokens(gold)
    if not g:
        return 0
    return int(tokens(pred) == g)


def contains(pred: str, gold: str) -> int:
    """1 se a resposta de referência (normalizada) aparece inteira na previsão."""
    g = " ".join(tokens(gold))
    return int(bool(g) and f" {g} " in f" {' '.join(tokens(pred))} ")
