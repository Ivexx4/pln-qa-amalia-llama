#!/usr/bin/env python3
"""Ponto de entrada. Exemplos:
    python main.py generate --limit 5
    python main.py all
    python main.py rag corpus --theme "Futebol"      # Meta 3 (RAG)
    python main.py rag all --theme "Futebol"
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rag":
        from qa_pipeline.rag_cli import main
        main(sys.argv[2:])
    else:
        from qa_pipeline.cli import main
        main()
