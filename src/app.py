"""
GPT-Style Chat UI — Streamlit app with sidebar session management.
"""

import streamlit as st
import os
import sys
import uuid
import time
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.ingest import PDFProcessor
from src.vector_store import VectorStore
from src.graph import run_agent
from src.memory import MemoryManager, list_all_sessions, delete_session_data

# ------------------------------------------------------------------ #
#  Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Custom CSS — dark GPT-style theme
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ---- New Chat Button ---- */
    .new-chat-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 15px;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .new-chat-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* ---- Session Cards ---- */
    .session-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .session-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(102, 126, 234, 0.4);
    }

    .session-card.active {
        background: rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }

    .session-name {
        font-size: 14px;
        font-weight: 500;
        color: #e0e0e0;
        margin: 0;
    }

    .session-id-text {
        font-size: 11px;
        color: #888 !important;
        margin: 4px 0 0 0;
        font-family: 'Courier New', monospace;
    }

    /* ---- Chat Area ---- */
    .chat-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    .chat-header h1 {
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .chat-header p {
        color: #888;
        font-size: 13px;
        margin: 4px 0 0 0;
    }

    /* ---- Welcome screen ---- */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 80px 20px;
        text-align: center;
    }

    .welcome-icon {
        font-size: 64px;
        margin-bottom: 20px;
    }

    .welcome-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
    }

    .welcome-subtitle {
        color: #888;
        font-size: 15px;
        max-width: 500px;
        line-height: 1.6;
    }

    /* ---- Status badges ---- */
    .memory-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: #667eea;
        margin: 2px;
    }

    /* ---- Divider ---- */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Session state initialization
# ------------------------------------------------------------------ #
if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # {session_id: {name, messages}}

if "active_session" not in st.session_state:
    st.session_state.active_session = None

if "memory_managers" not in st.session_state:
    st.session_state.memory_managers = {}  # {session_id: MemoryManager}

# Load existing sessions from disk on first run
if "sessions_loaded" not in st.session_state:
    st.session_state.sessions_loaded = True
    existing = list_all_sessions()
    for s in existing:
        sid = s["session_id"]
        if sid not in st.session_state.sessions:
            mm = MemoryManager(sid)
            st.session_state.memory_managers[sid] = mm
            st.session_state.sessions[sid] = {
                "name": s.get("name", f"Chat {sid[:6]}"),
                "messages": mm.get_messages(),
            }


def create_new_session():
    session_id = str(uuid.uuid4())[:8]
    name = f"Chat {time.strftime('%H:%M')}"
    st.session_state.sessions[session_id] = {
        "name": name,
        "messages": [],
    }
    mm = MemoryManager(session_id)
    mm.save_meta(name)
    st.session_state.memory_managers[session_id] = mm
    st.session_state.active_session = session_id


def switch_session(session_id: str):
    st.session_state.active_session = session_id


# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    # ---- New Chat Button ----
    st.markdown("")
    if st.button("➕  New Chat", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- Debug Info (Temporary) ----
    with st.expander("🛠️ Debug Config"):
        st.write(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
        st.write(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
        key = os.getenv('LANGCHAIN_API_KEY')
        st.write(f"Key set: {'Yes' if key else 'No'}")
        if key:
            st.write(f"Key contents: {key[:5]}...{key[-5:] if len(key)>5 else ''}")


    # ---- PDF Upload ----
    st.markdown("##### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.button("⚡ Process Documents", use_container_width=True):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Set GROQ_API_KEY first.")
        else:
            with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                try:
                    vs = VectorStore()
                    total_chunks = 0
                    
                    for uploaded_file in uploaded_files:
                        target_path = f"temp_{uploaded_file.name}"
                        with open(target_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            f.write =lenrhsts 

                        processor = PDFProcessor(target_path)
                        content = processor.extract_content()
                        
                        # Add to vector store
                        vs.add_documents(content)
                        total_chunks += len(content)

                        if os.path.exists(target_path):
                            os.remove(target_path)
                            
                    st.success(f"✅ Indexed {len(uploaded_files)} documents ({total_chunks} chunks) successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ---- Clear Database Button ----
    if st.button("🗑️ Clear Database", use_container_width=True):
        try:
            vs = VectorStore()
            vs.clear_collection()
            st.success("✅ Database cleared! Old documents removed.")
        except Exception as e:
            st.error(f"Error clearing: {e}")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- Chat Sessions ----
    st.markdown("##### 💬 Chat Sessions")

    if not st.session_state.sessions:
        st.caption("No chats yet. Click '+ New Chat' to start.")
    else:
        for sid, data in list(st.session_state.sessions.items()):
            is_active = (sid == st.session_state.active_session)
            icon = "🔵" if is_active else "⚪"
            label = f"{icon} {data['name']}"

            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                if st.button(
                    label,
                    key=f"session_{sid}",
                    use_container_width=True,
                    type="secondary" if not is_active else "primary",
                ):
                    switch_session(sid)
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{sid}", help="Delete this chat"):
                    delete_session_data(sid)
                    del st.session_state.sessions[sid]
                    if st.session_state.active_session == sid:
                        st.session_state.active_session = None
                    st.rerun()

            # Show session ID below
            # st.caption(f"`{sid}`")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ---- API Key ----
    st.markdown("##### ⚙️ Settings")
    if not os.getenv("GROQ_API_KEY"):
        api_key = st.text_input("Groq API Key", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("Key set!")
    else:
        st.success("🔑 API Key active")

    # ---- Memory info ----
    if st.session_state.active_session:
        mm = st.session_state.memory_managers.get(st.session_state.active_session)
        if mm:
            st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
            st.markdown("##### 🧠 Memory Status")
            n_short = len(mm.short_term) // 2
            n_long = len(mm.long_term_facts)
            st.markdown(
                f'<span class="memory-badge">Short-term: {n_short} turns</span> '
                f'<span class="memory-badge">Long-term: {n_long} facts</span>',
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------------ #
#  Main Chat Area
# ------------------------------------------------------------------ #
active = st.session_state.active_session

if active is None:
    # Welcome screen
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🧠</div>
        <div class="welcome-title">Document Intelligence</div>
        <div class="welcome-subtitle">
            Upload a PDF and ask questions — powered by LangGraph with RAG routing,
            short-term & long-term memory. Click <b>+ New Chat</b> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    session_data = st.session_state.sessions[active]
    mm = st.session_state.memory_managers.get(active)

    # Header
    st.markdown(
        f'<div class="chat-header">'
        f'<h1>{session_data["name"]}</h1>'
        f'<p>Session: {active}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Display messages
    for msg in session_data["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about your document or general questions..."):
        # Add user message
        session_data["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if mm is None:
                        mm = MemoryManager(active)
                        st.session_state.memory_managers[active] = mm

                    result = run_agent(prompt, mm)
                    response = result["response"]
                    tool_used = result["tool_used"]

                    # Show tool usage
                    if "RAG" in tool_used:
                        st.markdown(f"🛠️ *{tool_used}*")
                    else:
                        st.markdown(f"🧠 *{tool_used}*")
                    
                    st.markdown(response)

                    session_data["messages"].append(
                        {"role": "assistant", "content": response}
                    )

                    # Auto-rename session based on first message
                    if len(session_data["messages"]) == 2:
                        short_name = prompt[:30] + ("..." if len(prompt) > 30 else "")
                        session_data["name"] = short_name
                        mm.save_meta(short_name)

                except Exception as e:
                    st.error(f"Error: {e}")
