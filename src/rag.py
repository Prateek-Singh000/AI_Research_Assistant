import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

class RAGPipeline:
    def __init__(self, data_dir="data", use_deepseek=False):
        self.data_dir = data_dir
        # Use local HuggingFace embeddings (no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def load_documents(self):
        documents = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for file in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path)
            else:
                continue
            documents.extend(loader.load())
        return documents

    def build_vector_store(self):
        docs = self.load_documents()
        if not docs:
            print("No documents found in data/ folder.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local("faiss_index")
        print(f"Indexed {len(chunks)} chunks.")
        return self.vector_store

    def load_vector_store(self):
        if os.path.exists("faiss_index"):
            self.vector_store = FAISS.load_local(
                "faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return self.vector_store

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            self.load_vector_store()
        if self.vector_store is None:
            return []
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]