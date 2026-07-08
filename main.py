from dotenv import load_dotenv
import logging
import argparse
import os

os.environ["NNPACK_DISABLE"] = "1"

from src.utils import setup_logging
from src.extraction import Document_Reader
from src.indexing import Document_Indexer
from src.retrieval import Content_Retriever

DOCS_DIR = "docs"

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

def resolve_filepath(files: list) -> list:
    resolved = []
    for file in files:
        if os.path.exists(file):
            resolved.append(file)
        else:
            docs_path = os.path.join(DOCS_DIR, file)
            if os.path.exists(docs_path):
                resolved.append(docs_path)
            else:
                logger.warning(f"Arquivo não encontrado: '{file}'. Ignorando.")
    return resolved

    

def cmd_index(args):
    reader = Document_Reader()
    indexer = Document_Indexer()

    extracted = []
    files = resolve_filepath(args.files)

    for path in files:
        extracted.append(reader.extract_text(path))
    for url in args.urls:
        extracted.append(reader.extract_text_from_url(url))

    indexer.process_and_index(extracted, index_path=args.index)
    print(f"\níndice criado com sucesso em '{args.index}'.")

def cmd_ask(args):
    retriever = Content_Retriever(
        provider=args.provider,
        index_path=args.index
    )
    result = retriever.retrieve_and_answer(args.question)

    print(f"\nResposta: {result['response']}")
    print(f"\nFontes: {', '.join(result['sources'])}")
    print(f"Provedor: {result['provider']}")

def main():
    parser = argparse.ArgumentParser(description="RAG Document Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # Comando: indexar
    p_index = sub.add_parser("index", help="Indexa documentos")
    p_index.add_argument("--files", nargs="*", default=[], help="Caminhos de PDFs ou imagens")
    p_index.add_argument("--urls",  nargs="*", default=[], help="URLs para scraping")
    p_index.add_argument("--index", default="faiss_index", help="Pasta do índice FAISS")
    p_index.set_defaults(func=cmd_index)

    # Comando: consultar
    p_ask = sub.add_parser("ask", help="Faz uma pergunta ao pipeline")
    p_ask.add_argument("question", help="Pergunta em linguagem natural")
    p_ask.add_argument("--index",    default="faiss_index", help="Pasta do índice FAISS")
    p_ask.add_argument("--provider", default=None, help="Provedor de LLM: groq, gemini, ollama")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()