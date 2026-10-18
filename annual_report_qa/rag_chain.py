"""
RAG Chain for Annual Report Q&A Assistant.
Steps 5 and 6 of the workflow:
Step 5: Retrieve top-k relevant chunks for a question
Step 6: Generate a grounded answer with Gemini using the financial prompt
"""

import sys
from typing import List, Dict, Any, Optional

import config
from ingest import get_vectorstore

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


def format_context_with_citations(docs: List[Document]) -> str:
    """Add [Source | Page] markers so the LLM can cite exact pages."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Report")
        page = doc.metadata.get("page", "Unknown")
        formatted.append(f"[Source: {source} | Page: {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted)


def extract_content_text(content_obj: Any) -> str:
    """
    Gemini can return a list of content parts instead of a plain string.
    This pulls out only the readable text (skips thinking blocks / signatures).
    """
    if isinstance(content_obj, str):
        return content_obj

    if isinstance(content_obj, list):
        text_parts = []
        for item in content_obj:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif "text" in item and item.get("type") != "thinking":
                    text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            return "\n".join(text_parts)

    return str(content_obj)


def get_llm(model_name: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model (low temperature for factual answers)."""
    selected_model = model_name or config.DEFAULT_GEMINI_MODEL
    api_key = config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

    return ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=api_key,
        temperature=0.1,
        max_retries=1,  # fail fast so the fallback model can take over
    )


class FinancialRAGChain:
    """End-to-end RAG chain for querying indexed annual reports."""

    def __init__(
        self,
        vectorstore=None,
        model_name: Optional[str] = None,
        top_k: int = config.TOP_K_CHUNKS,
    ):
        self.vectorstore = vectorstore or get_vectorstore()
        self.primary_model_name = model_name or config.DEFAULT_GEMINI_MODEL
        self.top_k = top_k
        self.prompt_template = PromptTemplate(
            template=config.FINANCIAL_QA_PROMPT,
            input_variables=["context", "question"],
        )

    def retrieve(
        self,
        question: str,
        source_filter: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Step 5: top-k similarity search, optionally limited to one PDF."""
        search_k = k or self.top_k
        if source_filter:
            return self.vectorstore.similarity_search(
                query=question, k=search_k, filter={"source": source_filter}
            )
        return self.vectorstore.similarity_search(query=question, k=search_k)

    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
        k: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Step 5 + 6: retrieve, build the prompt, generate a grounded answer."""
        retrieved_docs = self.retrieve(question, source_filter=source_filter, k=k)

        if not retrieved_docs:
            return {
                "question": question,
                "answer": "No relevant context found in the indexed reports. Please upload or index an annual report first.",
                "citations": [],
                "sources": [],
                "context_docs": [],
                "model_used": "none",
            }

        formatted_prompt = self.prompt_template.format(
            context=format_context_with_citations(retrieved_docs),
            question=question,
        )

        # Primary model first, then fallbacks if rate-limited (429) or unavailable (503)
        candidate_models = [model_override or self.primary_model_name]
        for fallback in ["gemma-4-26b-a4b-it", "gemini-3.7-flash", "gemini-flash-latest"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        answer_text = ""
        model_used = candidate_models[0]

        for model in candidate_models:
            try:
                response = get_llm(model_name=model).invoke(formatted_prompt)
                answer_text = extract_content_text(response.content)
                model_used = model
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                if any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                    continue
                raise

        if not answer_text and last_error:
            raise last_error

        # Unique (source, page) citations
        citations = []
        seen = set()
        for doc in retrieved_docs:
            page = doc.metadata.get("page")
            source = doc.metadata.get("source", "Report")
            if (source, page) not in seen:
                seen.add((source, page))
                citations.append({
                    "page": page,
                    "source": source,
                    "snippet": doc.page_content[:250].strip() + "...",
                })

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "sources": sorted({c["page"] for c in citations if c.get("page") is not None}),
            "context_docs": retrieved_docs,
            "model_used": model_used,
        }


def ask_report(
    question: str,
    source_filter: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper to query the chain in one call."""
    return FinancialRAGChain(model_name=model_name).query(question, source_filter=source_filter)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What was the total revenue in FY2024?"
    print(f"\nQuestion: {q}")
    result = FinancialRAGChain().query(q)
    print(f"\nModel Used: {result['model_used']}")
    print(f"Citations: Pages {result['sources']}")
    print(f"\nAnswer:\n{result['answer']}")
