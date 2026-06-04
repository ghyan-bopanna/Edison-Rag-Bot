"""
RAG Pipeline - Core Classes
Extracted from notebook/pdf_loader.ipynb
"""

import os
import uuid
import numpy as np
from typing import List, Any
from pathlib import Path

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# 1. PDF Loader
# ─────────────────────────────────────────────

def process_all_pdfs(pdf_directory: str) -> List[Any]:
    """Process all PDF files in a directory and return LangChain documents."""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        return []

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
        except Exception:
            pass

    return all_documents


# ─────────────────────────────────────────────
# 2. Chunking
# ─────────────────────────────────────────────

def chunk_documents(
    documents: List[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Any]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────
# 3. Embedding Manager
# ─────────────────────────────────────────────

class EmbeddingManager:
    """Wraps a SentenceTransformer model to generate dense embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode(texts, show_progress_bar=False)


# ─────────────────────────────────────────────
# 4. Vector Store (ChromaDB)
# ─────────────────────────────────────────────

class VectorStore:
    """Manages document embeddings in a ChromaDB persistent vector store."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "./data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )

    def count(self) -> int:
        return self.collection.count()

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )

    def clear(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )


# ─────────────────────────────────────────────
# 5. RAG Retriever
# ─────────────────────────────────────────────

class RAGRetriever:
    """Retrieves relevant documents from the vector store using cosine similarity."""

    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManager):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[dict]:
        query_embedding = self.embedding_manager.generate_embeddings([query])

        results = self.vectorstore.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.vectorstore.count()),
        )

        if not results or not results["ids"][0]:
            return []

        retrieved_docs = []
        for doc_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity
            similarity_score = 1 - distance
            if similarity_score >= score_threshold:
                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance": distance,
                    }
                )

        return retrieved_docs


# ─────────────────────────────────────────────
# 6. Groq LLM
# ─────────────────────────────────────────────

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "qwen2-72b-instruct",
]


class GroqLLM:
    """Wraps ChatGroq with a RAG-optimised prompt template."""

    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY in your .env file."
            )

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024,
        )

    def generate_response(self, query: str, context: str) -> str:
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so.""",
        )

        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


# ─────────────────────────────────────────────
# 7. End-to-end RAG helper
# ─────────────────────────────────────────────

def rag_answer(
    query: str,
    retriever: RAGRetriever,
    llm: GroqLLM,
    top_k: int = 5,
) -> tuple[str, List[dict]]:
    """Retrieve context and generate an LLM answer. Returns (answer, retrieved_docs)."""
    docs = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([d["content"] for d in docs]) if docs else ""

    if not context:
        return "No relevant context found to answer the question.", []

    answer = llm.generate_response(query, context)
    return answer, docs
