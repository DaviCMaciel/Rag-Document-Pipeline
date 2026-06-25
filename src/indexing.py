import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class DocumentIndexer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Detecta se há GPU disponível para acelerar os embeddings
        device = "cuda" if os.environ.get('COLAB_GPU', False) else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name, 
            model_kwargs={'device': device}
        )
        # O splitter agora é configurado uma vez no init
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
            add_start_index=True,
        )

    def process_and_index(self, extracted_data_list, index_path="faiss_index"):
        """
        Recebe uma lista de dicionários vinda do Document_Reader:
        [{'content': '...', 'metadata': {'source': '...'}}, ...]
        """
        print(f"\nIniciando chunking de {len(extracted_data_list)} documentos...")
        
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
            print(f"-> {data['metadata']['source']}: {len(chunks)} chunks gerados.")

        if not langchain_docs:
            print("Erro: Nenhum conteúdo válido para indexar.")
            return None

        print(f"\nCriando índice FAISS com {len(langchain_docs)} chunks...")
        vector_store = FAISS.from_documents(langchain_docs, self.embeddings)
        
        # Salva localmente para não precisar reprocessar tudo toda hora
        vector_store.save_local(index_path)
        print(f"Índice salvo com sucesso em: {index_path}")
        
        return vector_store