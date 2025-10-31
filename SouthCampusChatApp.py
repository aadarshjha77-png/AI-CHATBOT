# app_select_index_with_openai_key.py
import os
import streamlit as st
from pathlib import Path
from typing import Dict

# ===================== SET YOUR OPENAI KEY HERE =====================
os.environ["OPENAI_API_KEY"] = 
# ====================================================================

# ----------------- CONFIG -----------------
BASE_INDEX_DIR = Path(r"C:\Users\aadar\Documents\pdf\FaissIndex")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"   # you can change to "gpt-4o" etc.
# ------------------------------------------

# ---------- LOGO & TITLE (robust, centered) ----------
BASE_DIR = Path(__file__).parent   # ensures path is relative to the script file
LOGO_PATH = BASE_DIR / "assets" / "du_logo.png"

# set page config first (must be before any other st.* calls)
if LOGO_PATH.exists():
    st.set_page_config(page_title="AI ASSISTANT (DU)", layout="wide", page_icon=str(LOGO_PATH))
else:
    st.set_page_config(page_title="AI ASSISTANT (DU)", layout="wide", page_icon="🎓")

# show helpful debug info (optional)
if not LOGO_PATH.exists():
    st.warning(f"DU logo not found at: {LOGO_PATH}\nPlace the file 'du_logo.png' inside the folder: {BASE_DIR / 'assets'}")
else:
    # Create a centered header: use columns with three middle columns; put logo+title in center column
    left, center, right = st.columns([0.9, 1, 1])
    with center:
        # put image and title stacked and centered
        st.image(str(LOGO_PATH), width=300)
       
st.title("AI ASSISTANT FOR SOUTH CAMPUS (DU)")

# ---- session state: chat history & config ----
if "history" not in st.session_state:
    # history is list of {"role": "user"|"assistant", "text": str, "sources": list}
    st.session_state.history = []

# how many previous exchanges to include in prompt
MAX_HISTORY = 6  # number of previous turns to include (change as needed)
# -----------------------------------------------


# -------- Robust imports (works across LangChain versions) ----------
CommunityFAISS = None
# --- Latest LangChain v0.3+ imports (no fallbacks) ---
# --- Modern LangChain v0.3+ imports ---
# Modern LangChain imports (v0.3+)

# Robust import block: prefer new langchain_community, but fallback to old langchain if needed
try:
    from langchain_community.vectorstores import FAISS as CommunityFAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    print("Using langchain_community imports")
except Exception:
    # fallback to older monolithic langchain layout
    try:
        from langchain.vectorstores import FAISS as CommunityFAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.chat_models import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        print("Using legacy langchain imports")
    except Exception as e:
        raise ImportError(
            "Could not import required LangChain modules. "
            "Please install langchain-community (or legacy langchain) in the environment running Streamlit.\n"
            f"Underlying error: {e}"
        )

# --------------------------------------
# -----------------------------------------------------

# Discover FAISS indexes
@st.cache_data(ttl=300)
def discover_indexes(base_dir: Path) -> Dict[str, Path]:
    """
    Scan base_dir for subfolders that contain a 'faiss_index' folder.
    Returns a mapping: folder_name -> path_to_faiss_index_folder
    Example:
      "Department (Biophysics and Bioinformatics).merged_header_tuples" ->
      "C:\\Users\\aadar\\Documents\\pdf\\FaissIndex\\Department (Biophysics and Bioinformatics).merged_header_tuples\\faiss_index"
    """
    mapping = {}
    if not base_dir.exists():
        return mapping

    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        faiss_folder = sub / "faiss_index"
        if faiss_folder.exists() and faiss_folder.is_dir():
            mapping[sub.name] = faiss_folder

    return mapping

index_map = discover_indexes(BASE_INDEX_DIR)
if not index_map:
    st.warning(f"No FAISS indexes found under {BASE_INDEX_DIR.resolve()}.")
    st.stop()

# Sidebar
st.sidebar.header("Index / Retrieval settings")
selected_name = st.sidebar.selectbox("Choose FAISS DB", options=list(index_map.keys()))
k = st.sidebar.number_input("Top-k retrieved passages", value=TOP_K, min_value=1, max_value=20, step=1)
use_llm = st.sidebar.checkbox("Use OpenAI LLM", value=True)
openai_model_name = st.sidebar.text_input("OpenAI model name", value=DEFAULT_OPENAI_MODEL)

selected_index_path = index_map[selected_name]
st.sidebar.markdown(f"**Selected path:** `{selected_index_path}`")


# Load embeddings & FAISS
@st.cache_resource
def load_store(index_path: str, model_name: str):
    # index_path should point to the folder that contains the saved index files
    p = Path(index_path)
    if not p.exists():
        raise FileNotFoundError(f"Index path does not exist: {index_path}")

    # instantiate embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to create HuggingFaceEmbeddings('{model_name}'): {e}")

    # ensure CommunityFAISS is available
    if CommunityFAISS is None:
        raise RuntimeError("CommunityFAISS is not available. Check that langchain-community is installed.")

    # load the FAISS index
    try:
        # some langchain-community versions accept allow_dangerous_deserialization kw
        try:
            store = CommunityFAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)
        except TypeError:
            store = CommunityFAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index from '{index_path}': {e}")

    return store

with st.spinner("Loading FAISS index and embeddings..."):
    store = load_store(str(selected_index_path), EMBEDDING_MODEL)

st.success(f"Loaded index: **{selected_name}**")

query = st.text_input("Ask anything about the selected documents:")

if query:
    retriever = store.as_retriever(search_kwargs={"k": k})

    # Step 1 — Retrieve relevant documents
    with st.spinner("Retrieving relevant passages..."):
        # use the vectorstore directly — synchronous and simple
        retrieved_docs = store.similarity_search(query, k=k)


    # Combine top documents into a context string
    context_texts = []
    for i, d in enumerate(retrieved_docs, start=1):
        meta = d.metadata or {}
        title = meta.get("source_file", "Unknown source")
        page = meta.get("page", "?")
        context_texts.append(f"[Source {i} | {title} | page {page}]\n{d.page_content}")

    context = "\n\n".join(context_texts)

    

    # Step 3 — Ask OpenAI to synthesize answer
    if os.getenv("OPENAI_API_KEY"):
        st.subheader("AI Assistant Answer (based on retrieved context)")

        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3
        )
        system_prompt = (
            "You are a knowledgeable and professional academic assistant. "
            "Your task is to answer user questions **only** using the information found in the provided context below.\n\n"
            "### Response Requirements:\n"
            "1. Begin with a brief, friendly introduction that summarizes what the answer will cover.\n"
            "2. Present the main answer in a clear, structured, and factual manner, citing relevant points from the context.\n"
            "3. Avoid adding external knowledge that is not explicitly supported by the provided documents.\n"
            "4. If the context does not contain the answer, clearly say: "
            "'The information is not available in the provided documents.'\n"
            "5. End your response with a concise conclusion or closing remark that summarizes the key takeaway.\n\n"
            "### Tone & Style:\n"
            "- Maintain a professional, informative, and neutral tone.\n"
            "- Write in full sentences with correct grammar and clarity.\n"
            "- Do not repeat the question unless it improves readability.\n\n"
            "### Example format:\n"
            "**Introduction:** A short overview of what the answer will explain.\n"
            "**Answer:** Well-structured explanation supported by context.\n"
            "**Conclusion:** A brief summary or closing statement.\n\n"
            "Use this structure for every answer."
        )


        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]

        with st.spinner("Generating answer..."):
            response = llm.invoke(messages)

        st.write(response.content.strip())

    else:
        st.warning("No OPENAI_API_KEY found. Please set your key to generate an answer.")

# ---- footer ----
st.markdown(
    """
    <div style="
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 5px 0;
        background-color: #f0f2f6;
        border-top: 0px solid #d3d3d3;
        font-size: 14px;
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    ">
        <span style="font-size:12px;">🎓</span>
        <b>Developed at South Campus by Institute of Informatics & Communication</b> 
        <span style="font-size:12px;">📘</span><br>
    
    </div>
    """,
    unsafe_allow_html=True,
)
# ----------------

