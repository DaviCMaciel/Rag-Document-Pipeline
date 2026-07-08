import os
import logging
import torch
from src.utils import get_device
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

class Content_Retriever:
    def __init__(self, provider: str = None, index_path: str = "faiss_index", model_name: str="sentence-transformers/all-MiniLM-L6-v2"):
        self.index_path = index_path

        self.provider = (provider if provider else os.getenv("LLM_PROVIDER", "groq")).lower()
        logger.info(f"Usando o provedor de LLM: {self.provider}")

        device = get_device()

        logger.info(f"Usando o dispositivo: {device} para embeddings.")
        self.embeddings = HuggingFaceEmbeddings(
            model_name = model_name, 
            model_kwargs = {'device': device}
        )

        if not os.path.exists(self.index_path):
            logger.error(f"Erro: O índice FAISS não foi encontrado no caminho especificado: {self.index_path}")
            raise FileNotFoundError(f"Índice FAISS não encontrado. Execute a indexação primeiro.")
        
        self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> Any:
        """Fábrica de modelos de linguagem. Inicializa a LLM com base no provedor ativo."""
        logger.info(f"Conectando à API/Serviço do provedor: '{self.provider}'")
        
        try:
            if self.provider == "groq":
                from langchain_groq import ChatGroq
                logger.info("Inicializando o modelo GroqChat...")
                return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
            
            elif self.provider == "ollama":
                from langchain_community.chat_models import ChatOllama
                # Modelo local leve e eficiente para tarefas de RAG
                return ChatOllama(model="llama3.2:3b", temperature=0.1)
            
            elif self.provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
            
            else:
                logger.warning(f"Provedor '{self.provider}' não é suportado oficialmente. Aplicando fallback automático para 'groq'.")
                from langchain_groq import ChatGroq
                return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        
        except ImportError as e:
            logger.error(f"Erro ao importar o módulo do provedor '{self.provider}': {e}")
            logger.info("Certifique-se de que os pacotes necessários estão instalados no seu ambiente virtual.")
            raise e
    
    def _format_docs(self, docs: List[Any]) -> Dict[str, Any]:
        """Agrupa o conteúdo dos documentos recuperados em uma única string formatada."""
        return "\n\n".join([doc.page_content for doc in docs])
    
    def retrieve_and_answer(self, query: str) -> str:
        """Recupera documentos relevantes e gera uma resposta baseada neles."""
        logger.info(f"Recebendo consulta: '{query}'")

        relevant_docs = self.retriever.invoke(query)
        sources = list(set(doc.metadata.get("source", "Desconhecido") for doc in relevant_docs))

        template = """
        Você é um assistente técnico especializado e focado em precisão.
        Responda à pergunta do usuário utilizando APENAS as informações contidas no contexto abaixo.
        Se a resposta não puder ser formulada com base no contexto, responda exatamente: "Não encontrei essa informação nos documentos fornecidos".

        Contexto:
        {context}

        Pergunta:
        {question}

        Resposta:
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": lambda x: self._format_docs(relevant_docs), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        
        return {
            "query": query,
            "response": response,
            "sources": sources,
            "provider": self.provider
        } 
