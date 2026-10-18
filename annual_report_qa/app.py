"""
Streamlit Web Application for Annual Report Q&A Assistant.
- PDF upload & automatic ingestion into ChromaDB
- Source citations with exact page numbers
- Multi-document switcher
- Model selector (gemini-3.8-flash default)
- Chat memory & conversation history
- Context inspection accordion
- Compare Reports tab (side-by-side answers)
- Export chat as Markdown
"""

import time
from datetime import datetime

import streamlit as st

import config
from ingest import ingest_pdf, get_vectorstore, get_indexed_sources
from rag_chain import FinancialRAGChain

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Annual Report Q&A Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

if "compare_cache" not in st.session_state:
    st.session_state.compare_cache = {}


@st.cache_resource
def load_db():
    return get_vectorstore()


db = load_db()


# ---------------- Helpers ----------------
def export_chat_to_markdown(messages) -> str:
    """Turn the chat history into a Markdown document."""
    lines = [
        "# Annual Report Q&A - Chat Export",
        f"_Exported: {datetime.now():%Y-%m-%d %H:%M}_",
        "",
    ]
    for msg in messages:
        if msg["role"] == "user":
            lines += [f"## Q: {msg['content']}", ""]
        else:
            lines += [msg["content"], ""]
            citations = msg.get("citations") or []
            if citations:
                pages = ", ".join(f"Page {c['page']} ({c['source']})" for c in citations)
                lines += [f"**Sources:** {pages}", ""]
            if msg.get("model_used"):
                lines += [f"_Model: {msg['model_used']}_", ""]
    return "\n".join(lines)


def render_citation_badges(citations, extra_html: str = ""):
    badge_html = "".join(
        f'<span class="citation-badge">📄 Page {c["page"]} ({c["source"]})</span>'
        for c in citations
    )
    st.markdown(badge_html + extra_html, unsafe_allow_html=True)


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

    # What is already inside ChromaDB (the real source of truth)
    indexed_sources = get_indexed_sources(db)

    if uploaded_file is not None:
        target_path = config.DATA_DIR / uploaded_file.name

        # Saving the file and indexing it are two separate checks
        if not target_path.exists():
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")

        if uploaded_file.name not in indexed_sources:
            with st.spinner("Extracting pages, generating chunks & embedding into ChromaDB..."):
                t0 = time.time()
                try:
                    res = ingest_pdf(target_path, vectorstore=db, clear_existing_source=True)
                    dur = round(time.time() - t0, 1)
                    st.success(f"Indexed {res['total_pages']} pages ({res['total_chunks']} chunks) in {dur}s!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        else:
            st.info(f"'{uploaded_file.name}' is already indexed.")

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
        index=0,
        help="Default: gemini-3.8-flash, with automatic fallback to other models.",
    )

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=2,
        max_value=8,
        value=config.TOP_K_CHUNKS,
        step=1,
        help="Number of most relevant text chunks sent to the model as context.",
    )

    st.markdown("---")
    st.subheader("📊 Index Status")
    st.write(f"**Storage Directory:** `{config.CHROMA_DIR.name}/`")
    st.write(f"**Total Reports Indexed:** `{len(indexed_sources)}`")
    if indexed_sources:
        with st.expander("View Indexed Files"):
            for s in indexed_sources:
                st.write(f"• `{s}`")

    st.download_button(
        "📄 Export Chat",
        data=export_chat_to_markdown(st.session_state.messages),
        file_name="chat_export.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        try:
            all_data = db.get()
            if all_data and all_data.get("ids"):
                db.delete(ids=all_data["ids"])
            st.session_state.messages = []
            st.session_state.compare_cache = {}
            st.success("Database cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing database: {e}")


# ================= MAIN AREA =================
st.markdown('<div class="main-header">Annual Report Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Financial analysis grounded strictly in company filings with page-numbered source citations.</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_compare = st.tabs(["Single Report Chat", "Compare Reports"])

# ---------------- TAB 1: Single Report Chat ----------------
with tab_chat:
    with st.expander("ℹ️ How This Works (2-Phase Architecture)", expanded=False):
        st.markdown("""
        ```
        PDF Upload → Load Text (PyPDFLoader) → Chunk (1000/150) → Embed (Gemini) → ChromaDB
                                                                                      ↓
        User Question → Embed Query → Retrieve Top-K (k=4) → Prompt + Gemini 3.8 Flash → Answer + Citations
        ```
        - **Anti-Hallucination Rule:** Prompt enforces `Answer using ONLY the context. If not in context, say 'I could not find this in the report.'`
        - **Grounded Verification:** Every cited fact lists the exact page number from the filing.
        """)

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

    # Chat history (chat memory)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown("**Citations:**")
                render_citation_badges(msg["citations"])
            if msg.get("context_docs"):
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for idx, doc in enumerate(msg["context_docs"], 1):
                        st.markdown(f"**Chunk {idx} — {doc.get('source', 'Report')} (Page {doc.get('page', 'Unknown')}):**")
                        st.text(doc.get("content", ""))
                        st.markdown("---")

    # Input: chat box or quick-question button
    user_prompt = st.chat_input("Ask a question about the annual report...")
    if st.session_state.quick_question:
        user_prompt = st.session_state.quick_question
        st.session_state.quick_question = None

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

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
                    answer = res["answer"]
                    citations = res["citations"]
                    model_used = res["model_used"]

                    st.markdown(answer)

                    model_badge = f'<span class="model-badge">⚡ {model_used} • {latency}s</span>'
                    if citations:
                        st.markdown("**Citations:**")
                        render_citation_badges(citations, extra_html=model_badge)
                    else:
                        st.markdown(model_badge, unsafe_allow_html=True)

                    context_info = []
                    if res["context_docs"]:
                        with st.expander("🔍 View Retrieved Context Chunks"):
                            for idx, doc in enumerate(res["context_docs"], 1):
                                page = doc.metadata.get("page", "Unknown")
                                source = doc.metadata.get("source", "Report")
                                st.markdown(f"**Chunk {idx} — {source} (Page {page}):**")
                                st.text(doc.page_content)
                                st.markdown("---")
                                context_info.append({
                                    "page": page,
                                    "source": source,
                                    "content": doc.page_content,
                                })

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "context_docs": context_info,
                        "model_used": model_used,
                    })

                except Exception as e:
                    st.error(f"Error querying assistant: {e}")


# ---------------- TAB 2: Compare Reports ----------------
with tab_compare:
    st.subheader("Compare Two Reports")

    if len(indexed_sources) < 2:
        st.info("Index at least two reports (use the sidebar uploader) to compare them side by side.")
    else:
        pick_a, pick_b = st.columns(2)
        with pick_a:
            report_a = st.selectbox("Report A", options=indexed_sources, index=0, key="cmp_a")
        with pick_b:
            report_b = st.selectbox("Report B", options=indexed_sources, index=1, key="cmp_b")

        cmp_question = st.text_input("Ask a question to compare both reports:", key="cmp_question").strip()

        if cmp_question:
            if report_a == report_b:
                st.warning("Please choose two different reports.")
            else:
                rag = FinancialRAGChain(vectorstore=db, model_name=selected_model, top_k=top_k)
                col_a, col_b = st.columns(2)

                for column, report in ((col_a, report_a), (col_b, report_b)):
                    with column:
                        st.markdown(f"**Answer from {report}:**")
                        cache_key = (cmp_question, report, selected_model, top_k)

                        if cache_key not in st.session_state.compare_cache:
                            with st.spinner("Analyzing..."):
                                try:
                                    st.session_state.compare_cache[cache_key] = rag.query(
                                        question=cmp_question,
                                        source_filter=report,
                                        k=top_k,
                                        model_override=selected_model,
                                    )
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                    continue

                        result = st.session_state.compare_cache[cache_key]
                        st.markdown(result["answer"])
                        if result["citations"]:
                            st.markdown("**Citations:**")
                            for c in result["citations"]:
                                st.markdown(f"- Page {c['page']} ({c['source']})")
