"""RAG core: ingestion, retrieval, reranking, and generation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vector_db"

# Load environment variables from .env if present
load_dotenv()


def _format_docs(docs: Iterable) -> str:
    """Join document contents for prompting."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGService:
    def __init__(
        self,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
        llm_model: str | None = None,
        device: str | None = None,
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required to start the RAG service.")

        embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        reranker_model = reranker_model or os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-large"
        )
        llm_model = llm_model or os.getenv("OPENAI_MODEL", "gpt-4o")
        device = device or os.getenv("MODEL_DEVICE", "cpu")

        DATA_DIR.mkdir(exist_ok=True)
        VECTOR_DIR.mkdir(exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=self.embeddings,
            collection_name="knowledge_base",
        )

        self.reranker = HuggingFaceCrossEncoder(
            model_name=reranker_model, model_kwargs={"device": device}
        )

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            base_url=os.getenv("OPENAI_BASE_URL"),  # allow proxy endpoints
        )

        self.prompt = ChatPromptTemplate.from_template(
            (
                "You are a helpful assistant. Use the context to answer.\n"
                "If you are unsure, say you do not know.\n\n"
                "Context:\n{context}\n\nQuestion: {question}"
            )
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def ingest_document(self, file_path: str) -> int:
        """Load a document, split it, and persist embeddings."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        self.vector_store.add_documents(chunks)
        self.vector_store.persist()
        return len(chunks)

    def _rerank(self, question: str, docs: List) -> List:
        """Score docs with cross-encoder and return top-ranked list."""
        if not docs:
            return []

        pairs: List[Tuple[str, str]] = [(question, d.page_content) for d in docs]
        scores = list(self.reranker.score(pairs))
        ranked = sorted(zip(docs, scores), key=lambda t: t[1], reverse=True)
        return [doc for doc, _ in ranked]

    def get_answer(self, question: str, k: int = 10, top_n: int = 3) -> str:
        """Run retrieval + rerank + generation for a question."""
        docs = self.vector_store.similarity_search(question, k=k)
        if not docs:
            return "No documents are indexed yet. Please upload and ingest documents first."
        reranked = self._rerank(question, docs)[:top_n]
        context = _format_docs(reranked)
        return self.chain.invoke({"context": context, "question": question})


rag_service = RAGService()
