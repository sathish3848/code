"""
Streamlit Web Application for Annual Report Q&A Assistant.
Provides a modern financial chat interface with:
- PDF upload & automatic ingestion into ChromaDB
- Source citations with exact page numbers
- Multi-document switcher
- Model selector (gemini-3.8-flash default)
- Chat memory & conversation history
- Context inspection accordion
"""

import os
import sys
import time
from pathlib import Path
import streamlit as st

import config
from ingest import (
    ingest_pdf,
    get_vectorstore,
    get_indexed_sources,
)
from rag_chain import FinancialRAGChain

# Page configuration
st.set_page_config(
    page_title="Annual Report Q&A Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Financial Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
        border: 1px solid #BFDBFE;
    }
    .model-badge {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
        font-family: monospace;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "kpi_cache" not in st.session_state:
    st.session_state.kpi_cache = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = {}


@st.cache_resource
def load_db():
    return get_vectorstore()


import concurrent.futures

def get_kpi_snapshot(rag, source_filter):
    kpi_questions = {
        "Revenue": "What was the total revenue?",
        "Net Profit": "What was the net profit?",
        "EBIT": "What was the EBIT?",
        "EPS": "What was the earnings per share (EPS)?"
    }
    kpis = {}
    
    def fetch_kpi(label, q):
        try:
            res = rag.query(q, source_filter=source_filter, k=2)
            ans = res["answer"]
            return label, ans[:37] + "..." if len(ans) > 40 else ans
        except Exception:
            return label, "N/A"

    # Run all 4 queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_kpi, label, q) for label, q in kpi_questions.items()]
        for future in concurrent.futures.as_completed(futures):
            label, ans = future.result()
            kpis[label] = ans

    # Return in the original order
    return {k: kpis.get(k, "N/A") for k in kpi_questions.keys()}

db = load_db()

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bullish.png", width=64)
    st.title("Financial RAG Settings")
    st.caption("Powered by Gemini & ChromaDB")

    st.markdown("---")
    st.subheader("📁 1. Annual Report Upload")
    uploaded_file = st.file_uploader(
        "Upload Annual Report (PDF)",
        type=["pdf"],
        help="Upload a company's Annual Report or 10-K to index and query.",
    )

        # Document Selection (Feature: Multiple PDF Support)
    indexed_sources = get_indexed_sources(db)

    if uploaded_file is not None:
        target_path = config.DATA_DIR / uploaded_file.name
        if not target_path.exists():
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")

        prior = st.session_state.ingest_status.get(uploaded_file.name)
        if uploaded_file.name in indexed_sources:
            st.info(f"'{uploaded_file.name}' is already indexed.")
        elif prior and prior.get("status") == "error":
            st.error(prior.get("error", "This PDF could not be indexed."))
            if st.button("Retry ingest", key=f"retry_{uploaded_file.name}"):
                st.session_state.ingest_status.pop(uploaded_file.name, None)
                st.rerun()
        else:
            with st.spinner("Extracting pages, generating chunks & embedding into ChromaDB..."):
                t0 = time.time()
                try:
                    res = ingest_pdf(target_path, vectorstore=db, clear_existing_source=True)
                    dur = round(time.time() - t0, 1)
                    st.session_state.ingest_status[uploaded_file.name] = {
                        "status": "success",
                        "total_pages": res["total_pages"],
                        "total_chunks": res["total_chunks"],
                    }
                    st.success(
                        f"Indexed {res['total_pages']} pages ({res['total_chunks']} chunks) in {dur}s!"
                    )
                    st.rerun()
                except Exception as e:
                    st.session_state.ingest_status[uploaded_file.name] = {
                        "status": "error",
                        "error": str(e),
                    }
                    st.error(str(e))
            
    st.subheader("📑 2. Active Report")
    if indexed_sources:
        doc_options = ["All Indexed Reports"] + indexed_sources
        selected_doc = st.selectbox(
            "Select Target Report",
            options=doc_options,
            index=0,
            help="Filter your questions to a specific annual report or search across all.",
        )
        active_filter = None if selected_doc == "All Indexed Reports" else selected_doc
    else:
        st.warning("No documents indexed yet. Please upload a PDF or run ingest.")
        active_filter = None

    st.markdown("---")
    st.subheader("⚙️ 3. LLM & Retrieval")

    selected_model = st.selectbox(
        "Gemini Model",
        options=config.AVAILABLE_MODELS,
        index=0,  # Default to gemini-3.8-flash
        help="Target model: gemini-3.8-flash with fallback resilience.",
    )

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=2,
        max_value=8,
        value=config.TOP_K_CHUNKS,
        step=1,
        help="Number of most relevant text chunks to provide as context (Default: 4).",
    )

    st.markdown("---")
    st.subheader("📊 Index Status")
    st.write(f"**Storage Directory:** `{config.CHROMA_DIR.name}/`")
    st.write(f"**Total Reports Indexed:** `{len(indexed_sources)}`")
    if indexed_sources:
        with st.expander("View Indexed Files"):
            for s in indexed_sources:
                st.write(f"• `{s}`")

    def export_chat_to_markdown(messages):
        md_lines = []
        for msg in messages:
            if msg["role"] == "user":
                md_lines.append(f"## Q: {msg['content']}")
            elif msg["role"] == "assistant":
                md_lines.append(f"{msg['content']}")
                if msg.get("citations"):
                    sources = ", ".join([f"Page {c['page']} ({c['source']})" for c in msg["citations"]])
                    md_lines.append(f"\n**Sources:** {sources}")
                md_lines.append("\n---\n")
        return "\n".join(md_lines)

    if st.session_state.messages:
        md_str = export_chat_to_markdown(st.session_state.messages)
        st.download_button(
            label="📄 Export Chat",
            data=md_str,
            file_name="chat_export.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        try:
            # Delete all documents from chroma
            all_data = db.get()
            if all_data and all_data.get("ids"):
                db.delete(ids=all_data["ids"])
            st.session_state.messages = []
            st.success("Database cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing database: {e}")


# ================= MAIN CHAT AREA =================
st.markdown('<div class="main-header">Annual Report Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Financial analysis grounded strictly in company filings with page-numbered source citations.</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Single Report Chat", "Compare Reports"])

with tab1:
    # Pipeline Architecture Banner
    with st.expander("ℹ️ How This Works (2-Phase Architecture)", expanded=False):
        st.markdown("""
        ```
        PDF Upload → Load Text (PyMuPDF) → Chunk (1000/150) → Embed (Gemini) → ChromaDB
                                                                                      ↓
        User Question → Embed Query → Retrieve Top-K (k=4) → Prompt + Gemini 3.8 Flash → Answer + Citations
        ```
        - **Anti-Hallucination Rule:** Prompt enforces `Answer using ONLY the context. If not in context, say 'I could not find this in the report.'`
        - **Grounded Verification:** Every cited fact lists the exact page number from the filing.
        """)

    # Quick Question Buttons (Section 6 from Spec)
    st.write("**Suggested Finance Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("💰 FY2024 Revenue", use_container_width=True):
            st.session_state.quick_question = "What was the total revenue in FY2024?"
    with col2:
        if st.button("⚠️ Management Risks", use_container_width=True):
            st.session_state.quick_question = "What are the main risks mentioned by management?"
    with col3:
        if st.button("👥 Employee Count", use_container_width=True):
            st.session_state.quick_question = "How many employees does the company have?"
    with col4:
        if st.button("📈 Dividend per Share", use_container_width=True):
            st.session_state.quick_question = "What is the dividend per share?"


    if active_filter:
        if active_filter not in st.session_state.kpi_cache:
            with st.spinner(f"Generating KPIs for {active_filter}..."):
                # We need rag
                rag_temp = FinancialRAGChain(vectorstore=db, model_name=selected_model, top_k=top_k)
                st.session_state.kpi_cache[active_filter] = get_kpi_snapshot(rag_temp, active_filter)
        
        kpis = st.session_state.kpi_cache[active_filter]
        st.write(f"**KPI Snapshot:**")
        kpi_cols = st.columns(4)
        for idx, (label, value) in enumerate(kpis.items()):
            with kpi_cols[idx]:
                st.metric(label=label, value=value)
        st.markdown("---")
    # Render Chat History (Chat Memory feature)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                st.markdown("**Citations:**")
                badge_html = "".join([
                    f'<span class="citation-badge">📄 Page {c["page"]} ({c["source"]})</span>'
                    for c in msg["citations"]
                ])
                st.markdown(badge_html, unsafe_allow_html=True)

            if "context_docs" in msg and msg["context_docs"]:
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for idx, doc in enumerate(msg["context_docs"], 1):
                        p = doc.get("page", "Unknown")
                        s = doc.get("source", "Report")
                        c = doc.get("content", "")
                        st.markdown(f"**Chunk {idx} — {s} (Page {p}):**")
                        st.text(c)
                        st.markdown("---")


    # Process Input (either from quick question button or chat input box)
    user_prompt = st.chat_input("Ask a question about the annual report...")

    if st.session_state.quick_question:
        user_prompt = st.session_state.quick_question
        st.session_state.quick_question = None

    if user_prompt:
        # Append user question
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving report sections and analyzing context..."):
                t_start = time.time()
                rag = FinancialRAGChain(
                    vectorstore=db,
                    model_name=selected_model,
                    top_k=top_k,
                )

                try:
                    res = rag.query(
                        question=user_prompt,
                        source_filter=active_filter,
                        k=top_k,
                        model_override=selected_model,
                    )
                    latency = round(time.time() - t_start, 2)
                    ans = res["answer"]
                    citations = res["citations"]
                    model_used = res["model_used"]

                    # Display answer
                    st.markdown(ans)

                    # Display Page Citations
                    if citations:
                        st.markdown("**Citations:**")
                        badge_html = "".join([
                            f'<span class="citation-badge">📄 Page {c["page"]} ({c["source"]})</span>'
                            for c in citations
                        ])
                        badge_html += f'<span class="model-badge">⚡ {model_used} • {latency}s</span>'
                        st.markdown(badge_html, unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="model-badge">⚡ {model_used} • {latency}s</span>', unsafe_allow_html=True)

                    # Context Inspector Accordion
                    context_info = []
                    if res["context_docs"]:
                        with st.expander("🔍 View Retrieved Context Chunks"):
                            for idx, doc in enumerate(res["context_docs"], 1):
                                p = doc.metadata.get("page", "Unknown")
                                s = doc.metadata.get("source", "Report")
                                st.markdown(f"**Chunk {idx} — {s} (Page {p}):**")
                                st.text(doc.page_content)
                                st.markdown("---")
                                context_info.append({
                                    "page": p,
                                    "source": s,
                                    "content": doc.page_content,
                                })

                    # Persist in session state for chat memory
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "citations": citations,
                        "context_docs": context_info,
                        "model_used": model_used,
                    })

                except Exception as e:
                    st.error(f"Error querying assistant: {e}")

with tab2:
    st.subheader("Compare Two Reports")
    if len(indexed_sources) >= 2:
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            report_a = st.selectbox("Report A", options=indexed_sources, key="report_a")
        with comp_col2:
            report_b = st.selectbox("Report B", options=indexed_sources, key="report_b", index=1)
        
        compare_q = st.text_input("Ask a question to compare both reports:")
        if compare_q:
            col_a, col_b = st.columns(2)
            rag = FinancialRAGChain(vectorstore=db, model_name=selected_model, top_k=top_k)
            
            with col_a:
                st.markdown(f"**Answer from {report_a}:**")
                with st.spinner(f"Querying {report_a}..."):
                    try:
                        res_a = rag.query(compare_q, source_filter=report_a, k=top_k, model_override=selected_model)
                        st.markdown(res_a["answer"])
                        if res_a["citations"]:
                            st.markdown("**Citations:**")
                            for c in res_a["citations"]:
                                st.markdown(f"- Page {c['page']} ({c['source']})")
                    except Exception as e:
                        st.error(f"Error: {e}")

            with col_b:
                st.markdown(f"**Answer from {report_b}:**")
                with st.spinner(f"Querying {report_b}..."):
                    try:
                        res_b = rag.query(compare_q, source_filter=report_b, k=top_k, model_override=selected_model)
                        st.markdown(res_b["answer"])
                        if res_b["citations"]:
                            st.markdown("**Citations:**")
                            for c in res_b["citations"]:
                                st.markdown(f"- Page {c['page']} ({c['source']})")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Please index at least 2 reports to use the comparison feature.")
