"""
Ingestion Pipeline for Annual Report Q&A Assistant.
Implements Steps 1 to 4 of the workflow:
Step 1: Load PDF using PyMuPDF (fallback: pypdf) with page-number metadata
Step 2: Chunk text using RecursiveCharacterTextSplitter (chunk size 1000, overlap 150)
Step 3: Create embeddings using Gemini embeddings (models/gemini-embedding-001)
Step 4: Store vectors in ChromaDB (persisted to local directory chroma_db/)
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import config

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# pypdf prints "Error -3 while decompressing data" and "incorrect startxref"
# on damaged PDFs. Keep that noise off the Streamlit terminal.
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

try:
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
except Exception:
    pymupdf = None  # type: ignore

UTF8_REPLACEMENT = b"\xef\xbf\xbd"
MIN_PAGE_CHARS = 20


def get_embedding_function(model_name: Optional[str] = None):
    """
    Initializes and returns the embedding function for ChromaDB.
    Defaults to Gemini embeddings (gemini-embedding-001).
    """
    model = model_name or config.DEFAULT_EMBEDDING_MODEL
    api_key = config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file or environment variables."
        )
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
    )


def get_vectorstore(
    persist_directory: Optional[Path] = None,
    collection_name: str = "annual_reports",
    embedding_function=None,
) -> Chroma:
    """
    Returns an instance of ChromaDB connected to the persistent store.
    """
    persist_dir = persist_directory or config.CHROMA_DIR
    embeddings = embedding_function or get_embedding_function()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def _annotate_pages(pages: List[Document], path_obj: Path) -> List[Document]:
    """Store 1-indexed page numbers for human-readable citations."""
    for doc in pages:
        try:
            page_num = int(doc.metadata.get("page", 0))
        except (TypeError, ValueError):
            page_num = 0
        doc.metadata["page"] = page_num + 1
        doc.metadata["source"] = path_obj.name
        doc.metadata["file_path"] = str(path_obj.resolve())
    return pages


def _pages_with_text(pages: List[Document]) -> List[Document]:
    return [doc for doc in pages if (doc.page_content or "").strip() and len(doc.page_content.strip()) >= MIN_PAGE_CHARS]


def _corruption_message(path_obj: Path, total_pages: int) -> str:
    raw = path_obj.read_bytes()
    replacement_count = raw.count(UTF8_REPLACEMENT)
    if replacement_count > 500:
        return (
            f"Loaded {total_pages} page(s) from '{path_obj.name}' but extracted 0 usable text. "
            f"The file is binary-corrupted ({replacement_count:,} UTF-8 replacement bytes inside compressed PDF streams). "
            "That usually happens when a PDF is opened or saved as text (or copied through a text-only channel). "
            "Re-download the original binary .pdf and upload that file. "
            "A working text PDF (for example a native annual report, not this damaged copy) will index normally."
        )
    return (
        f"Loaded {total_pages} page(s) from '{path_obj.name}' but extracted 0 usable text. "
        "pypdf/PyMuPDF could not decompress page streams (zlib 'incorrect header check'). "
        "Use a text-based PDF, or re-export/print the file to a new PDF from the original source."
    )


def _load_with_pymupdf(path_obj: Path) -> List[Document]:
    if pymupdf is None:
        return []
    pages: List[Document] = []
    with pymupdf.open(str(path_obj)) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": page.number, "source": path_obj.name},
                )
            )
    return _annotate_pages(pages, path_obj)


def _load_with_pypdf(path_obj: Path) -> List[Document]:
    import os
    from contextlib import redirect_stderr
    from pypdf import PdfReader

    pages: List[Document] = []
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
        reader = PdfReader(str(path_obj), strict=False)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": i, "source": path_obj.name},
                )
            )
    return _annotate_pages(pages, path_obj)


def load_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Step 1: Load PDF text per page.
    Uses PyMuPDF first (more tolerant of broken xref tables), then pypdf.
    Empty pages are dropped. Raises if the file has no extractable text.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found at: {path_obj}")

    pages = _load_with_pymupdf(path_obj)
    usable = _pages_with_text(pages)

    # Skip pypdf on files already known to have destroyed Flate streams —
    # it only reprints zlib "incorrect header check" for every object.
    replacement_count = path_obj.read_bytes().count(UTF8_REPLACEMENT)
    if not usable and replacement_count <= 500:
        fallback = _load_with_pypdf(path_obj)
        if fallback:
            pages = fallback
        usable = _pages_with_text(fallback)

    if not usable:
        raise ValueError(_corruption_message(path_obj, len(pages)))

    print(f"      Extracted text from {len(usable)}/{len(pages)} page(s).")
    return usable


def chunk_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Step 2: Chunk text using RecursiveCharacterTextSplitter.
    Default parameters: chunk size 1000, overlap 150.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    vectorstore: Optional[Chroma] = None,
    clear_existing_source: bool = False,
) -> Dict[str, Any]:
    """
    Full Indexing Pipeline (Steps 1 to 4):
    1. Load PDF
    2. Chunk text
    3. Embed chunks
    4. Store into ChromaDB
    """
    path_obj = Path(pdf_path)
    filename = path_obj.name

    print(f"\n[1/4] Loading PDF: {filename}...")
    pages = load_pdf(path_obj)
    total_pages = len(pages)
    print(f"      Loaded {total_pages} page(s) with extractable text.")

    print(f"[2/4] Chunking document (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
    chunks = chunk_documents(pages)
    total_chunks = len(chunks)
    print(f"      Generated {total_chunks} chunk(s).")
    if total_chunks == 0:
        raise ValueError(
            f"'{filename}' produced 0 chunks after splitting. "
            "The PDF has no usable text to embed."
        )

    print(f"[3/4 & 4/4] Creating embeddings and storing into ChromaDB...")
    db = vectorstore or get_vectorstore()

    # If requested or replacing, clear existing chunks for this specific document source
    if clear_existing_source:
        try:
            # Query existing IDs for this source and delete
            existing_results = db.get(where={"source": filename})
            if existing_results and existing_results.get("ids"):
                db.delete(ids=existing_results["ids"])
                print(f"      Removed {len(existing_results['ids'])} previous chunks for {filename}.")
        except Exception as e:
            print(f"      Note: Could not clear existing source records: {e}")

    # Add chunks with metadata in batches to avoid rate limits
    import time
    batch_size = 50
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"      Persisting batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} ({len(batch)} chunks)...")
        try:
            db.add_documents(batch)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"      Rate limit hit on batch. Waiting 60 seconds before retrying...")
                time.sleep(60)
                db.add_documents(batch)
            else:
                raise e
    print(f"      Successfully persisted {total_chunks} chunks to {config.CHROMA_DIR}!")

    return {
        "filename": filename,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "status": "success",
    }


def ingest_all_pdfs(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Scans the data directory and ingests all PDF files found.
    """
    target_dir = data_dir or config.DATA_DIR
    pdf_files = list(target_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in directory: {target_dir}")
        return []

    print(f"Found {len(pdf_files)} PDF file(s) in {target_dir} to ingest.")
    results = []
    db = get_vectorstore()

    for pdf in pdf_files:
        try:
            res = ingest_pdf(pdf, vectorstore=db, clear_existing_source=True)
        except Exception as e:
            print(f"Failed to ingest {pdf.name}: {e}")
            res = {
                "filename": pdf.name,
                "total_pages": 0,
                "total_chunks": 0,
                "status": "error",
                "error": str(e),
            }
        results.append(res)

    return results


def get_indexed_sources(vectorstore: Optional[Chroma] = None) -> List[str]:
    """
    Returns a sorted list of unique PDF sources currently indexed in ChromaDB.
    """
    db = vectorstore or get_vectorstore()
    try:
        data = db.get()
        metadatas = data.get("metadatas", [])
        sources = {m.get("source") for m in metadatas if m and m.get("source")}
        return sorted(list(sources))
    except Exception:
        return []


if __name__ == "__main__":
    # If a specific PDF argument is passed via CLI, ingest that.
    # Otherwise ingest all PDFs in data/
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
        ingest_pdf(target_path, clear_existing_source=True)
    else:
        ingest_all_pdfs()
