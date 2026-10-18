import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enterprise_market_intel.crew import EnterpriseMarketIntelCrew

# Page Configuration
st.set_page_config(
    page_title="Gemini - Enterprise Market Intelligence", page_icon="✨", layout="wide"
)

# Custom CSS matching Gemini Dark Mode UI layout
st.markdown(
    """
<style>
    .stApp {
        background-color: #131314;
        color: #e3e3e3;
        font-family: 'Google Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #1e1f20;
        border-right: 1px solid #2d2e30;
    }
    [data-testid="stSidebar"] * {
        color: #c4c7c5 !important;
    }
    .stTextInput input {
        background-color: #1e1f20 !important;
        color: #e3e3e3 !important;
        border-radius: 28px !important;
        border: 1px solid #444746 !important;
        padding: 14px 20px !important;
    }
    .stTextInput input:focus {
        border-color: #a8c7fa !important;
        box-shadow: 0 0 0 1px #a8c7fa !important;
    }
    .stButton button {
        background-color: #d3e3fd;
        color: #001d35;
        border-radius: 24px;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        transition: background 0.2s;
    }
    .stButton button:hover {
        background-color: #aecbfa;
        color: #001d35;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 400;
        color: #e3e3e3;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #8e918f;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .report-container {
        background-color: #1e1f20;
        border: 1px solid #333537;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        color: #e3e3e3;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State for Dynamic History Tracking
if "history" not in st.session_state:
    st.session_state.history = []  # Stores list of dicts: {"topic": str, "report": str}
if "active_report_index" not in st.session_state:
    st.session_state.active_report_index = None

# Dynamic History Sidebar Bar
with st.sidebar:
    st.markdown("### ✨ Gemini")
    st.markdown("---")

    if st.button("💬 New Research", use_container_width=True):
        st.session_state.active_report_index = None
        st.rerun()

    st.markdown("---")
    st.markdown("**Recent Research History**")

    if not st.session_state.history:
        st.markdown(
            "<p style='color: #8e918f; font-size: 0.9rem;'>No past queries yet.</p>",
            unsafe_allow_html=True,
        )
    else:
        # Display history from newest to oldest
        for idx, item in enumerate(reversed(st.session_state.history)):
            orig_idx = len(st.session_state.history) - 1 - idx
            display_title = (
                item["topic"]
                if len(item["topic"]) <= 22
                else item["topic"][:22] + "..."
            )
            if st.button(
                f"📁 {display_title}", key=f"hist_{orig_idx}", use_container_width=True
            ):
                st.session_state.active_report_index = orig_idx
                st.rerun()

# Main Interface Header
st.markdown(
    '<div class="hero-title">Enterprise Market Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Multi-agent research and strategic intelligence powered by Gemini.</div>',
    unsafe_allow_html=True,
)

# Check whether we are viewing a historical report or running a new query
if st.session_state.active_report_index is not None:
    current_item = st.session_state.history[st.session_state.active_report_index]
    st.markdown(f"### 📊 Report: *{current_item['topic']}*")

    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(current_item["report"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        label="📥 Download Report as Markdown",
        data=current_item["report"],
        file_name=f"market_intel_{current_item['topic'].lower().replace(' ', '_')}.md",
        mime="text/markdown",
    )

    if st.button("⬅️ Start New Search"):
        st.session_state.active_report_index = None
        st.rerun()
else:
    # Input Area for New Search
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        topic = st.text_input(
            "Ask Gemini",
            value="",
            label_visibility="collapsed",
            placeholder="Enter research topic or industry trend...",
        )
        submitted = st.button("🚀 Run Research Crew", use_container_width=True)

    if submitted:
        if not topic.strip():
            st.warning("Please enter a valid research topic.")
        else:
            with st.spinner(f"🤖 Gemini agents are researching '{topic}'..."):
                try:
                    inputs = {"topic": topic}
                    result = EnterpriseMarketIntelCrew().crew().kickoff(inputs=inputs)
                    report_content = (
                        result.raw if hasattr(result, "raw") else str(result)
                    )

                    # Append result to history and set it as active
                    st.session_state.history.append(
                        {"topic": topic, "report": report_content}
                    )
                    st.session_state.active_report_index = (
                        len(st.session_state.history) - 1
                    )
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during execution: {e}")
