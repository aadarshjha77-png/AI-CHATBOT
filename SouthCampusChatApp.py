# ---- imports & config (replace your current imports block with this) ----
import os, time
import streamlit as st
from pathlib import Path
from typing import Dict

# LangChain (install: pip install -U langchain-community langchain-openai langchain-core sentence-transformers faiss-cpu)
from langchain_community.vectorstores import FAISS as CommunityFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# OpenAI key

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Basic Streamlit page setup (no logo)
st.set_page_config(page_title="AI ASSISTANT (DU)", layout="wide", page_icon="🤖")

# Paths / constants
BASE_INDEX_DIR = Path(r"C:\Users\aadar\Documents\pdf\FaissIndex")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
TOP_K = 4

# ---- Session state defaults (add this once, near the top) ----
if "mode" not in st.session_state: st.session_state.mode = "home"   # "home" | "chat"
if "query" not in st.session_state: st.session_state.query = ""
if "history" not in st.session_state: st.session_state.history = []  # list of {role, text}
if "typing" not in st.session_state: st.session_state.typing = False
if "typing_full" not in st.session_state: st.session_state.typing_full = ""
if "typing_pos" not in st.session_state: st.session_state.typing_pos = 0
if "skip" not in st.session_state: st.session_state.skip = False

# Put this near the top (after imports and set_page_config)
AVATAR_HTML = """
<div style="display:flex;justify-content:center;margin:10px 0 6px 0;">
  <svg width="125" height="140" viewBox="0 0 84 84" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="grad" cx="30%" cy="30%" r="80%">
        <stop offset="0%"  stop-color="#6EA8FE"/>
        <stop offset="60%" stop-color="#9A7DF5"/>
        <stop offset="100%" stop-color="#6F42C1"/>
      </radialGradient>
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="rgba(50,50,93,0.35)"/>
      </filter>
    </defs>
    <circle cx="42" cy="42" r="38" fill="url(#grad)" filter="url(#shadow)"/>
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
          font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif"
          font-weight="700" font-size="26" fill="#ffffff" letter-spacing="1">AI</text>
  </svg>
</div>
"""

def header():
    # render avatar
    st.markdown(AVATAR_HTML, unsafe_allow_html=True)
    # render title separately
    st.markdown(
        "<h2 style='text-align:center;margin:4px 0 12px 0;'>AI ASSISTANT FOR SOUTH CAMPUS (DU)</h2>",
        unsafe_allow_html=True
    )

def render_home():
    # render avatar first (NOT inside the CSS string)
    st.markdown(AVATAR_HTML, unsafe_allow_html=True)

    # now render the rest of the hero section
    st.markdown(
        """
        <style>
        .hero { padding: 2rem 1rem 0.5rem 1rem; text-align: center; }
        .hero h1 { font-size: 2.0rem; line-height: 1.2; margin: 0.4rem 0 0.6rem 0; }
        .sub { color:#60666d; margin-bottom:1.2rem; }
        .card { max-width: 820px; margin: 0.5rem auto; padding: 0.8rem 1rem;
                border: 1px solid #e6e8eb; background:#fafafa; border-radius:12px; }
        .chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:10px; }
        .chip { border:1px solid #e6e8eb; padding:6px 10px; border-radius:999px; background:white; font-size:13px; }
        </style>
        <div class="hero">
          <h1>How may I help you?</h1>
          <div class="sub">Ask about departments, courses, admissions, research areas, facilities, and more.</div>
        </div>
        """,
        unsafe_allow_html=True
    )


    with st.form("home_form", clear_on_submit=False):
        q = st.text_area(
            "Type your question",
            placeholder="e.g., Subjects in Semester 1 (Institute of Informatics & Communication)?",
            height=90
        )
        submitted = st.form_submit_button("Ask", use_container_width=True)
    if submitted and q.strip():
        st.session_state.mode = "chat"
        st.session_state.query = q.strip()
        st.rerun()

    # quick suggestions (no logos)
    cols = st.columns(4)
    sugg = [
        "Faculty list for Biophysics & Bioinformatics",
        "Research labs in South Campus",
        "Course structure for M.Sc. Genetics",
        "Admission criteria for IIC",
    ]
    for i, s in enumerate(sugg):
        with cols[i % 4]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state.mode = "chat"
                st.session_state.query = s
                st.rerun()


@st.cache_data(ttl=300)
def discover_indexes(base_dir:Path)->Dict[str,Path]:
    m={}
    if base_dir.exists():
        for s in sorted(base_dir.iterdir()):
            if s.is_dir() and (s/"faiss_index").exists():
                m[s.name]=s/"faiss_index"
    return m

index_map=discover_indexes(BASE_INDEX_DIR)
st.sidebar.header("Select Database")
if not index_map:
    st.sidebar.warning("No FAISS indexes found.")
    if st.session_state.mode=="home":
        render_home(); st.stop()
    else:
        st.stop()
selected_name=st.sidebar.selectbox("Choose FAISS DB",options=list(index_map.keys()))
selected_index_path=index_map[selected_name]
st.sidebar.markdown(f"**Selected:** `{selected_name}`")

st.sidebar.markdown("### Retrieval Settings")

deep_search = st.sidebar.toggle("Enable Deep Search 🔍", value=False)

if deep_search:
    k = 4   # retrieves more context
else:
    k = 2   # faster, short context


if st.session_state.mode=="home":
    render_home(); st.stop()

@st.cache_resource
def load_store(p:str):
    emb=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store=CommunityFAISS.load_local(p,emb,allow_dangerous_deserialization=True)
    return store

with st.spinner("Loading FAISS index..."):
    store=load_store(str(selected_index_path))
st.success(f"Loaded: {selected_name}")

# Chat CSS and container
st.markdown("""<style>
.chat-container{max-height:430px;overflow-y:auto;padding:12px;border:1px solid #dcdcdc;background:#fafafa;border-radius:8px}
.user-msg{background:#d9eafd;color:black;padding:8px 12px;border-radius:12px;margin:6px 0;width:fit-content;margin-left:auto;max-width:90%}
.bot-msg{background:#ececec;color:black;padding:8px 12px;border-radius:12px;margin:6px 0;width:fit-content;margin-right:auto;max-width:90%}
.send-row{display:flex;gap:8px}
</style>""",unsafe_allow_html=True)

st.markdown("### Chat")
st.markdown('<div class="chat-container">',unsafe_allow_html=True)

# render history except pending typing
for m in st.session_state.history:
    cls="user-msg" if m["role"]=="user" else "bot-msg"
    st.markdown(f'<div class="{cls}">{m["text"]}</div>',unsafe_allow_html=True)

# placeholder for typing bubble if active
placeholder = st.empty()
if st.session_state.typing:
    text_to_show = st.session_state.typing_full
    pos = st.session_state.typing_pos
    # show partial
    partial = " ".join(text_to_show.split()[0:pos]) if pos>0 else ""
    placeholder.markdown(f'<div class="bot-msg">{partial}</div>',unsafe_allow_html=True)
else:
    placeholder.empty()

st.markdown("</div>",unsafe_allow_html=True)

# Input area (multi-line) and Send button (SEND1 style: Send ✉️)
q = st.text_area("Your question", value=st.session_state.query, height=120, key="main_input")
st.session_state.query = q
cols = st.columns([1,0.2])
with cols[1]:
    send = st.button("Send ✉️",use_container_width=True)
# Process send
if send and st.session_state.query.strip():
    user_q = st.session_state.query.strip()
    st.session_state.history.append({"role":"user","text":user_q})
    st.session_state.query = ""  # clear input to avoid repeats
    # Retrieval
    with st.spinner("Retrieving..."):
        docs = store.similarity_search(user_q, k=TOP_K)
    chunks = [d.page_content.strip() for d in docs if d.page_content.strip()]
    if not chunks:
        ans = "The information is not available in the provided documents."
    else:
        context = "\n".join(chunks)
        system_prompt = ("Answer using only the information provided in the context. "
                         "Do not add intro or conclusion. "
                         "If multiple points, use bullet points with '•'. "
                         "If only one fact, reply in one plain sentence. "
                         "Do NOT mention sources.")
        final_prompt = f"Context:\n{context}\n\nQuestion: {user_q}\n\nAnswer:"
        llm = ChatOpenAI(model_name=DEFAULT_OPENAI_MODEL, temperature=0)
        resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=final_prompt)])
        ans = resp.content.strip()
    # start typing sequence
    st.session_state.typing=True
    st.session_state.typing_full=ans
    st.session_state.typing_pos=0
    st.session_state.skip=False
    # animate: reveal 3-5 words per step, medium speed ~0.04s
    words = ans.split()
    step_min,step_max=3,5
    i=0
    while i < len(words):
        if st.session_state.skip:
            # show full immediately
            placeholder.markdown(f'<div class="bot-msg">{ans}</div>',unsafe_allow_html=True)
            break
        take = step_min if len(words)-i>step_min else (len(words)-i)
        # adjust take to a small random-like pattern (fixed here to step_max when possible)
        take = step_max if len(words)-i>=step_max else (len(words)-i)
        i += take
        st.session_state.typing_pos = i
        partial = " ".join(words[0:i])
        placeholder.markdown(f'<div class="bot-msg">{partial}</div>',unsafe_allow_html=True)
        time.sleep(0.04)
    # finish: append final text to history, stop typing, clear placeholders
    st.session_state.history.append({"role":"assistant","text":ans})
    st.session_state.typing=False
    st.session_state.typing_full=""
    st.session_state.typing_pos=0
    placeholder.empty()
    st.rerun()

# Always show Skip Animation button for assistant messages during typing (SK2)
if st.session_state.typing:
    if st.button("Skip Animation ⏩"):
        st.session_state.skip=True
        st.rerun()

# New conversation and Home buttons
c1,c2 = st.columns([1,1])
with c1:
    if st.button("🧹 New Conversation"):
        st.session_state.mode="home"; st.session_state.query=""; st.session_state.history=[]; st.session_state.typing=False; st.session_state.typing_full=""; st.rerun()
with c2:
    if st.button("🏠 Home"):
        st.session_state.mode="home"; st.rerun()

# Footer
st.markdown("""<style>.footer{position:fixed;left:0;bottom:0;width:100%;text-align:center;padding:8px 0;background:#f0f2f6;border-top:1px solid #d3d3d3;font-size:14px;color:#333}.footer a{color:#004080;font-weight:600}.footer a:hover{color:#800000}</style><div class="footer">🎓 Developed at <a href="https://www.du.ac.in/" target="_blank">South Campus — University of Delhi</a> 📘<br><span style="color:#666">For internal academic use • Powered by AI 🤖</span></div>""",unsafe_allow_html=True)
