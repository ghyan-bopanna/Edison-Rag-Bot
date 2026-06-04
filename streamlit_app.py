"""
Edison RAG Bot — Streamlit UI
Upload PDFs → Ask questions → Get AI answers grounded in your documents.
Mascot: Edison, the RAG Bot 🤖
"""

import base64
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    AVAILABLE_MODELS,
    EmbeddingManager,
    GroqLLM,
    RAGRetriever,
    VectorStore,
    chunk_documents,
    process_all_pdfs,
    rag_answer,
)

# ── env & page config ──────────────────────────────────────────────────────────
load_dotenv()

def _get_groq_key() -> str:
    """Resolve API key: st.secrets (Streamlit Cloud) → .env (local)."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()

def _key_source() -> str:
    """Return a human-readable label for where the key came from."""
    try:
        if st.secrets.get("GROQ_API_KEY"):
            return "Streamlit Secrets"
    except Exception:
        pass
    return ".env"

st.set_page_config(
    page_title="Edison RAG Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── load Edison image as base64 ────────────────────────────────────────────────
_EDISON_PATH = Path(__file__).parent / "edison.png"

def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_EDISON_B64 = _img_b64(_EDISON_PATH) if _EDISON_PATH.exists() else ""
_EDISON_SRC = f"data:image/png;base64,{_EDISON_B64}" if _EDISON_B64 else ""

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }

    /* ── Animated grid overlay ── */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(52,199,89,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(52,199,89,0.04) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #010409 !important;
        border-right: 1px solid #1f6b2e !important;
    }
    [data-testid="stSidebar"] > div {
        padding-top: 0 !important;
    }

    /* ── Sidebar header ── */
    .sidebar-header {
        background: linear-gradient(180deg, #0d2818 0%, #010409 100%);
        border-bottom: 1px solid #1f6b2e;
        padding: 1.2rem 1rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sidebar-name {
        font-size: 1.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #34c759, #ffd60a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.3px;
    }
    .sidebar-tagline {
        font-size: 0.7rem;
        color: #57a55a;
        margin-top: 0.2rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #34c759;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1rem 0 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #1f6b2e, transparent);
    }

    /* ── Stat cards ── */
    .stat-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stat-card {
        flex: 1;
        background: #0d2818;
        border: 1px solid #1f6b2e;
        border-radius: 10px;
        padding: 0.7rem 0.6rem;
        text-align: center;
    }
    .stat-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #34c759;
        line-height: 1;
    }
    .stat-lbl {
        font-size: 0.62rem;
        color: #57a55a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* ── Status pills ── */
    .pill-ready {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #0d2818; border: 1px solid #34c759;
        border-radius: 20px; padding: 0.25rem 0.75rem;
        font-size: 0.75rem; color: #34c759; font-weight: 600;
    }
    .pill-idle {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: #1a0a0a; border: 1px solid #f85149;
        border-radius: 20px; padding: 0.25rem 0.75rem;
        font-size: 0.75rem; color: #f85149; font-weight: 600;
    }
    .dot-pulse {
        width: 7px; height: 7px; border-radius: 50%;
        background: #34c759;
        animation: pulse 1.4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.8); }
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(52,199,89,0.3) !important;
    }

    /* ── Main hero ── */
    .hero-wrap {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 2rem 0 1rem;
    }
    .hero-mascot {
        width: 120px;
        filter: drop-shadow(0 0 20px rgba(52,199,89,0.45));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-6px); }
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(90deg, #34c759 0%, #ffd60a 60%, #34c759 100%);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        letter-spacing: -1px;
    }
    @keyframes shimmer {
        0%   { background-position: 0% }
        100% { background-position: 200% }
    }
    .hero-sub {
        color: #57a55a;
        font-size: 0.95rem;
        margin-top: 0.2rem;
    }

    /* ── Divider ── */
    hr { border-color: #1f6b2e !important; margin: 0.8rem 0 !important; }

    /* ── Chat bubbles ── */
    .bubble-user {
        background: linear-gradient(135deg, #1a3a22, #0d2818);
        border: 1px solid #2ea043;
        border-radius: 18px 18px 4px 18px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 78%;
        color: #e6edf3;
        font-size: 0.93rem;
        box-shadow: 0 4px 16px rgba(52,199,89,0.12);
    }
    .bubble-assistant {
        background: #010409;
        border: 1px solid #1f6b2e;
        border-radius: 18px 18px 18px 4px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 78%;
        color: #e6edf3;
        font-size: 0.93rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    /* ── Source cards ── */
    .src-card {
        background: #0d1117;
        border: 1px solid #1f6b2e;
        border-left: 3px solid #34c759;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin: 0.35rem 0;
        font-size: 0.81rem;
    }
    .src-badge {
        display: inline-block;
        background: #0d2818;
        border: 1px solid #2ea043;
        border-radius: 20px;
        padding: 0.1rem 0.5rem;
        font-size: 0.7rem;
        color: #34c759;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .src-file {
        color: #ffd60a;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .src-preview {
        color: #7d8590;
        line-height: 1.5;
        margin-top: 0.3rem;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
    }
    .empty-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #3d444d;
        margin-bottom: 0.3rem;
    }
    .empty-sub {
        font-size: 0.83rem;
        color: #30363d;
    }

    /* ── Env key badge ── */
    .env-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: #0d2818; border: 1px solid #2ea043;
        border-radius: 8px; padding: 0.4rem 0.7rem;
        font-size: 0.75rem; color: #34c759;
        margin-bottom: 0.6rem;
        width: 100%;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: #010409 !important;
        border: 1px solid #1f6b2e !important;
        border-radius: 14px !important;
        color: #e6edf3 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #34c759 !important;
        box-shadow: 0 0 0 3px rgba(52,199,89,0.15) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #1f6b2e !important;
        border-radius: 10px !important;
        background: #010409 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #1f6b2e; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #34c759; }

    /* ── Sidebar inputs ── */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #0d1117 !important;
        border-color: #1f6b2e !important;
        color: #e6edf3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "pipeline_ready": False,
        "embedding_manager": None,
        "vectorstore": None,
        "retriever": None,
        "groq_llm": None,
        "indexed_files": [],
        "doc_count": 0,
        "chunk_count": 0,
        "top_k": 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_embedding_manager():
    return EmbeddingManager()


def _rebuild_pipeline(
    pdf_dir: str,
    groq_api_key: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> tuple[bool, str]:
    """Load PDFs, embed, index, and initialise the retriever + LLM."""
    try:
        em = _get_embedding_manager()
        st.session_state.embedding_manager = em

        docs = process_all_pdfs(pdf_dir)
        if not docs:
            return False, "No PDF files found in the selected directory."

        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        vs = VectorStore(persist_directory=os.path.join(pdf_dir, ".vector_store"))
        if vs.count() > 0:
            vs.clear()

        embeddings = em.generate_embeddings([c.page_content for c in chunks])
        vs.add_documents(chunks, embeddings)

        retriever = RAGRetriever(vs, em)
        llm = GroqLLM(model_name=model_name, api_key=groq_api_key)

        st.session_state.vectorstore = vs
        st.session_state.retriever = retriever
        st.session_state.groq_llm = llm
        st.session_state.pipeline_ready = True
        st.session_state.doc_count = len(docs)
        st.session_state.chunk_count = len(chunks)
        st.session_state.indexed_files = list(
            {Path(d.metadata.get("source_file", "unknown")) for d in docs}
        )
        st.session_state.top_k = top_k

        return True, f"✅ Indexed **{len(chunks)} chunks** from **{len(docs)} pages** across **{len(st.session_state.indexed_files)} files**."
    except Exception as e:
        return False, f"❌ {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Header ──
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-name">Edison RAG Bot</div>
            <div class="sidebar-tagline">⚡ Powered by Groq · LangChain · FAISS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Status pill ──
    if st.session_state.pipeline_ready:
        st.markdown(
            '<div style="padding:0.4rem 1rem 0;">'
            '<span class="pill-ready"><span class="dot-pulse"></span> Pipeline Ready</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding:0.4rem 1rem 0;">'
            '<span class="pill-idle">● Not Indexed</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── API Key ──
    st.markdown('<div class="section-label">🔑 Groq API Key</div>', unsafe_allow_html=True)
    _resolved_key = _get_groq_key()
    if _resolved_key:
        st.markdown(
            f'<div class="env-badge">✅ &nbsp;API key loaded from <code>{_key_source()}</code></div>',
            unsafe_allow_html=True,
        )
        api_key_input = _resolved_key
    else:
        api_key_input = st.text_input(
            "Groq API Key",
            value="",
            type="password",
            label_visibility="collapsed",
            placeholder="gsk_...",
            key="api_key_input",
        )

    # ── Model ──
    st.markdown('<div class="section-label">🤖 Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        label_visibility="collapsed",
        key="model_select",
    )

    # ── PDF Source ──
    st.markdown('<div class="section-label">📂 PDF Source</div>', unsafe_allow_html=True)
    source_mode = st.radio(
        "Source",
        ["Upload files", "Use local folder"],
        label_visibility="collapsed",
        key="source_mode",
        horizontal=True,
    )

    uploaded_files = []
    pdf_folder = ""

    if source_mode == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
        )
    else:
        pdf_folder = st.text_input(
            "Folder path",
            value="",
            label_visibility="collapsed",
            placeholder="/path/to/your/pdf/folder",
            key="pdf_folder",
        )

    # ── Advanced ──
    st.markdown('<div class="section-label">⚙️ Settings</div>', unsafe_allow_html=True)
    with st.expander("Advanced Settings", expanded=False):
        chunk_size    = st.slider("Chunk size",    300, 2000, 1000, 100, key="chunk_size")
        chunk_overlap = st.slider("Chunk overlap",   0,  500,  200,  50, key="chunk_overlap")
        top_k         = st.slider("Top-K results",   1,   10,    5,   1, key="top_k_slider")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Index button ──
    index_btn = st.button(
        "⚡ Index Documents",
        use_container_width=True,
        type="primary",
        key="index_btn",
    )

    if index_btn:
        groq_key = api_key_input.strip() if isinstance(api_key_input, str) else ""
        if not groq_key:
            st.error("Please enter your Groq API key.")
        else:
            with st.spinner("Edison is reading your docs… 📖"):
                if source_mode == "Upload files":
                    if not uploaded_files:
                        st.error("Please upload at least one PDF.")
                    else:
                        tmp_dir = tempfile.mkdtemp()
                        for f in uploaded_files:
                            dest = os.path.join(tmp_dir, f.name)
                            with open(dest, "wb") as fh:
                                fh.write(f.read())
                        ok, msg = _rebuild_pipeline(
                            tmp_dir, groq_key, selected_model,
                            chunk_size, chunk_overlap, top_k,
                        )
                        if ok: st.success(msg)
                        else:  st.error(msg)
                else:
                    if not os.path.isdir(pdf_folder):
                        st.error("Folder not found.")
                    else:
                        ok, msg = _rebuild_pipeline(
                            pdf_folder, groq_key, selected_model,
                            chunk_size, chunk_overlap, top_k,
                        )
                        if ok: st.success(msg)
                        else:  st.error(msg)

    # ── Stats ──
    if st.session_state.pipeline_ready:
        st.markdown('<div class="section-label">📊 Index Stats</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-num">{st.session_state.doc_count}</div>
                    <div class="stat-lbl">Pages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{st.session_state.chunk_count}</div>
                    <div class="stat-lbl">Chunks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{len(st.session_state.indexed_files)}</div>
                    <div class="stat-lbl">Files</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.indexed_files:
            with st.expander("📋 Indexed Files"):
                for f in sorted(st.session_state.indexed_files):
                    st.markdown(f"• `{f}`")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero header (Edison appears here only) ──
st.markdown(
    f"""
    <div class="hero-wrap">
        <img class="hero-mascot" src="{_EDISON_SRC}" alt="Edison" />
        <div>
            <div class="hero-title">Edison RAG Bot</div>
            <div class="hero-sub">Ask Edison anything about your documents — grounded answers, no hallucinations.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Status bar ──
col_l, col_r = st.columns([5, 1])
with col_r:
    if st.session_state.pipeline_ready:
        st.markdown(
            '<div style="text-align:right;">'
            '<span class="pill-ready"><span class="dot-pulse"></span> Ready</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:right;">'
            '<span class="pill-idle">● Index docs first</span>'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Chat history ──
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div style="font-size:3rem; margin-bottom:0.8rem;">💬</div>
            <div class="empty-title">Hi! I'm Edison 👋</div>
            <div class="empty-sub">
                Index your PDFs using the sidebar, then ask me anything about them.<br>
                I'll find the most relevant passages and give you a grounded answer!
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-assistant">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 {len(sources)} source chunks used by Edison", expanded=False):
                    for i, s in enumerate(sources, 1):
                        score_pct = f"{s['similarity_score'] * 100:.1f}%"
                        src_file  = s["metadata"].get("source_file", "unknown")
                        page      = s["metadata"].get("page", "?")
                        preview   = s["content"][:280].replace("\n", " ") + "…"
                        st.markdown(
                            f"""
                            <div class="src-card">
                                <span class="src-badge">#{i}</span>
                                <span class="src-badge">{score_pct} match</span>
                                <span class="src-file">📄 {src_file} &nbsp;·&nbsp; page {page}</span>
                                <div class="src-preview">{preview}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ── Chat input ──
user_input = st.chat_input(
    "Ask Edison something about your documents…",
    key="chat_input",
    disabled=not st.session_state.pipeline_ready,
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Edison is thinking… ⚡"):
        try:
            answer, sources = rag_answer(
                user_input,
                st.session_state.retriever,
                st.session_state.groq_llm,
                top_k=st.session_state.get("top_k", 5),
            )
        except Exception as e:
            answer  = f"Oops! Something went wrong: {str(e)}"
            sources = []

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()
