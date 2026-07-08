from src.utils import get_device
import logging
from typing import List, Dict, Any
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class Document_Indexer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=100):
        self.model_name = model_name

        device = get_device()

        logger.info(f"Usando o dispositivo: {device} para embeddings.")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name, 
            model_kwargs={'device': device}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            add_start_index=True,
        )

    def process_and_index(self, extracted_data_list: List[Dict[str, Any]], index_path: str = "faiss_index") -> FAISS:
        """
        Recebe uma lista de dicionários vinda do Document_Reader:
        """
        logger.info(f"Iniciando chunking de {len(extracted_data_list)} documentos...")
        
        langchain_docs = []
        for data in extracted_data_list:
            if not data["content"].strip():
                continue
                
            # Converte o dicionário bruto em um objeto Document do LangChain
            doc = Document(
                page_content=data["content"],
                metadata=data["metadata"]
            )
            
            # Divide o documento em pedaços (chunks)
            chunks = self.text_splitter.split_documents([doc])
            langchain_docs.extend(chunks)
            logger.info(f"Documento '{data['metadata']['source']}' dividido em {len(chunks)} chunks.")

        if not langchain_docs:
            logger.error("Erro: Nenhum conteúdo válido para indexar.")
            raise ValueError("Falha ao indexar: Lista de documentos vazia.")

        logger.info(f"Criando índice FAISS com {len(langchain_docs)} chunks...")
        vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        
        # Salva localmente para não precisar reprocessar tudo toda hora
        vector_store.save_local(index_path)
        logger.info(f"Índice salvo com sucesso em: {index_path}")
        
        return vector_store