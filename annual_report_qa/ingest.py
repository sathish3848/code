"""
Ingestion Pipeline for Annual Report Q&A Assistant.
Steps 1 to 4 of the workflow:
Step 1: Load PDF using PyPDFLoader (metadata includes page numbers)
Step 2: Chunk text using RecursiveCharacterTextSplitter (size 1000, overlap 150)
Step 3: Create embeddings using Gemini embeddings
Step 4: Store vectors in ChromaDB (persisted to the local chroma_db/ folder)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import config

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# How many chunks to embed per API call (keeps big reports within rate limits)
EMBED_BATCH_SIZE = 100


def get_embedding_function(model_name: Optional[str] = None):
    """Create the Gemini embedding function used by ChromaDB."""
    model = model_name or config.DEFAULT_EMBEDDING_MODEL
    api_key = config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file."
        )
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def get_vectorstore(
    persist_directory: Optional[Path] = None,
    collection_name: str = "annual_reports",
    embedding_function=None,
) -> Chroma:
    """Return a ChromaDB instance connected to the persistent store."""
    persist_dir = persist_directory or config.CHROMA_DIR
    embeddings = embedding_function or get_embedding_function()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def load_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Step 1: Load PDF page by page.
    PyPDFLoader pages are 0-indexed; we convert to 1-indexed for citations.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at: {path_obj}")

    pages = PyPDFLoader(str(path_obj)).load()

    for doc in pages:
        raw_page = doc.metadata.get("page", 0)
        doc.metadata["page"] = int(raw_page) + 1
        doc.metadata["source"] = path_obj.name
        doc.metadata["file_path"] = str(path_obj.resolve())

    return pages


def chunk_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """Step 2: Split pages into overlapping chunks (page metadata is kept)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_pdf(
    pdf_path: str | Path,
    vectorstore: Optional[Chroma] = None,
    clear_existing_source: bool = False,
) -> Dict[str, Any]:
    """Full indexing pipeline: load -> chunk -> embed -> store."""
    path_obj = Path(pdf_path)
    filename = path_obj.name

    print(f"\n[1/4] Loading PDF: {filename}...")
    pages = load_pdf(path_obj)
    total_pages = len(pages)
    print(f"      Loaded {total_pages} page(s).")

    print(f"[2/4] Chunking (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
    chunks = chunk_documents(pages)
    total_chunks = len(chunks)
    print(f"      Generated {total_chunks} chunk(s).")

    print("[3/4 & 4/4] Creating embeddings and storing into ChromaDB...")
    db = vectorstore or get_vectorstore()

    # Remove old chunks of the same file so re-ingesting never creates duplicates
    if clear_existing_source:
        try:
            existing = db.get(where={"source": filename})
            if existing and existing.get("ids"):
                db.delete(ids=existing["ids"])
                print(f"      Removed {len(existing['ids'])} previous chunks for {filename}.")
        except Exception as e:
            print(f"      Note: could not clear existing records: {e}")

    for start in range(0, total_chunks, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        db.add_documents(batch)
        print(f"      Embedded {min(start + EMBED_BATCH_SIZE, total_chunks)}/{total_chunks} chunks")

    print(f"      Persisted {total_chunks} chunks to {config.CHROMA_DIR}")

    return {
        "filename": filename,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "status": "success",
    }


def ingest_all_pdfs(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Ingest every PDF found in the data/ folder."""
    target_dir = data_dir or config.DATA_DIR
    pdf_files = list(target_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {target_dir}")
        return []

    print(f"Found {len(pdf_files)} PDF file(s) in {target_dir}.")
    db = get_vectorstore()
    return [ingest_pdf(pdf, vectorstore=db, clear_existing_source=True) for pdf in pdf_files]


def get_indexed_sources(vectorstore: Optional[Chroma] = None) -> List[str]:
    """Return a sorted list of unique PDF filenames currently indexed."""
    db = vectorstore or get_vectorstore()
    try:
        data = db.get()
        metadatas = data.get("metadatas", [])
        sources = {m.get("source") for m in metadatas if m and m.get("source")}
        return sorted(sources)
    except Exception:
        return []


if __name__ == "__main__":
    # uv run python ingest.py            -> ingest every PDF in data/
    # uv run python ingest.py file.pdf   -> ingest one specific PDF
    if len(sys.argv) > 1:
        ingest_pdf(Path(sys.argv[1]), clear_existing_source=True)
    else:
        ingest_all_pdfs()
