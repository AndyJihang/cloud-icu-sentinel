"""Load medical guidelines into Qdrant for local RAG experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Settings, get_settings
from src.core.logger import configure_logging


def _list_guideline_files(knowledge_base_dir: Path) -> list[Path]:
    """Return markdown guideline files from the configured knowledge base directory."""

    if not knowledge_base_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {knowledge_base_dir}")

    guideline_files = sorted(path for path in knowledge_base_dir.glob("*.md") if path.is_file())
    if not guideline_files:
        raise FileNotFoundError(f"No markdown guideline files found in: {knowledge_base_dir}")

    return guideline_files


def _build_documents(guideline_files: Iterable[Path]) -> list[Document]:
    """Split guideline markdown files into chunked LangChain documents."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    documents: list[Document] = []

    for guideline_path in guideline_files:
        content = guideline_path.read_text(encoding="utf-8")
        raw_chunks = splitter.split_text(content)
        condition_name = guideline_path.stem.replace("_", " ")

        documents.extend(
            Document(
                page_content=chunk,
                metadata={
                    "source": str(guideline_path),
                    "file_name": guideline_path.name,
                    "condition": condition_name,
                    "chunk_index": index,
                    "document_type": "clinical_guideline",
                },
            )
            for index, chunk in enumerate(raw_chunks)
        )

    return documents


def build_vector_index() -> None:
    """Chunk all local guideline markdown files and upsert embeddings into Qdrant."""

    settings: Settings = get_settings()
    logger = configure_logging(settings)
    knowledge_base_dir: Path = settings.knowledge_base_dir

    guideline_files = _list_guideline_files(knowledge_base_dir)

    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required to build embeddings.")

    logger.info(
        "Loading %s guideline file(s) from %s",
        len(guideline_files),
        knowledge_base_dir,
    )
    documents = _build_documents(guideline_files)

    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_embedding_model,
    )

    logger.info(
        "Upserting %s chunks into Qdrant collection '%s'",
        len(documents),
        settings.qdrant_collection_name,
    )
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
        collection_name=settings.qdrant_collection_name,
        force_recreate=True,
    )

    logger.info(
        "Vector index build complete. Collection '%s' is ready.",
        settings.qdrant_collection_name,
    )


if __name__ == "__main__":
    build_vector_index()
