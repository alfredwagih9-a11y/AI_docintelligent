"""
AI Document Intelligence System  (RAG + Output Parser)
100% free & local: Ollama (LLM) + HuggingFace embeddings + FAISS + Streamlit
"""
import os
import re
import tempfile
from typing import List

import streamlit as st
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------
# 1. Output schemas (Output Parser)
# ----------------------------------------------------------------------------
class DocAnalysis(BaseModel):
    document: str = Field(description="Name(s) of the document(s) analysed")
    summary: str = Field(description="Concise summary or direct answer to the question")
    key_points: List[str] = Field(description="Most important points, short bullets")
    risks: List[str] = Field(description="Risks, red flags, obligations or penalties found. Empty list if none")
    references: List[str] = Field(description="Supporting evidence as 'file, page N: short paraphrase'")


class Comparison(BaseModel):
    documents: str = Field(description="Names of the two documents compared")
    summary: str = Field(description="Overall comparison in 3-5 sentences")
    similarities: List[str] = Field(description="What both documents share")
    differences: List[str] = Field(description="Key differences between them")
    risks: List[str] = Field(description="Risks or conflicts that arise from the differences")
    references: List[str] = Field(description="Supporting evidence as 'file, page N: short paraphrase'")


# ----------------------------------------------------------------------------
# 2. Prompts
# ----------------------------------------------------------------------------
SYSTEM = (
    "You are a careful document analyst. Use ONLY the provided context. "
    "If the answer is not in the context, say so in the summary. "
    "Always cite the source file and page in 'references'. "
    "Return ONLY valid JSON that follows the format instructions. No extra text.\n\n"
    "{format_instructions}"
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Document: {name}\n\nContext (sampled excerpts):\n{context}\n\n"
              "Task: Summarize the document, list key points and identify risks "
              "(legal, financial, deadlines, penalties, ambiguous clauses)."),
])

COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "=== DOCUMENT A: {a} ===\n{ctx_a}\n\n=== DOCUMENT B: {b} ===\n{ctx_b}\n\n"
              "Focus: {focus}\n\nTask: Compare the two documents."),
])


# ----------------------------------------------------------------------------
# 3. RAG helpers: loading, splitting, vector DB
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(model: str):
    return ChatOllama(model=model, temperature=0, format="json", num_ctx=8192)


def build_index(files):
    """Load PDFs -> split -> embed -> FAISS vector store."""
    chunks, names = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        pages = PyPDFLoader(path).load()
        os.unlink(path)
        for p in pages:
            p.metadata["source"] = f.name
            p.metadata["page"] = int(p.metadata.get("page", 0)) + 1
        chunks.extend(splitter.split_documents(pages))
        names.append(f.name)
    if not chunks:
        raise ValueError("No text found. Scanned PDFs need OCR.")
    store = FAISS.from_documents(chunks, get_embeddings())
    return store, chunks, names


def fmt(docs):
    return "\n\n".join(
        f"[{d.metadata['source']}, page {d.metadata['page']}]\n{d.page_content}" for d in docs
    )


def sample_chunks(chunks, name, max_chars=9000):
    """Evenly sample chunks across a document so the whole doc is represented."""
    own = [c for c in chunks if c.metadata["source"] == name]
    if not own:
        return []
    budget = max(1, max_chars // 1000)
    step = max(1, len(own) // budget)
    return own[::step][:budget]


def run_chain(prompt, parser, llm, **inputs):
    """prompt | llm | parser with a regex fallback if the model adds extra text."""
    inputs["format_instructions"] = parser.get_format_instructions()
    raw = (prompt | llm).invoke(inputs).content
    try:
        return parser.parse(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return parser.parse(m.group(0))
        raise


# ----------------------------------------------------------------------------
# 4. UI helpers
# ----------------------------------------------------------------------------
def show_list(title, items, empty="None found"):
    st.markdown(f"**{title}**")
    if items:
        for i in items:
            st.markdown(f"- {i}")
    else:
        st.caption(empty)


def show_analysis(r: DocAnalysis):
    st.subheader("📄 Document")
    st.write(r.document)
    st.subheader("📝 Summary")
    st.write(r.summary)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔑 Key Points")
        show_list("", r.key_points)
    with c2:
        st.subheader("⚠️ Risks")
        show_list("", r.risks, "No risks detected")
    st.subheader("📚 References")
    show_list("", r.references)


def show_comparison(r: Comparison):
    st.subheader("📄 Documents")
    st.write(r.documents)
    st.subheader("📝 Summary")
    st.write(r.summary)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🤝 Similarities")
        show_list("", r.similarities)
    with c2:
        st.subheader("↔️ Differences")
        show_list("", r.differences)
    st.subheader("⚠️ Risks")
    show_list("", r.risks, "No risks detected")
    st.subheader("📚 References")
    show_list("", r.references)


# ----------------------------------------------------------------------------
# 5. Streamlit app
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Document Intelligence", page_icon="📑", layout="wide")
st.title("📑 AI Document Intelligence System")
st.caption("RAG + Structured Output Parser • Free & local (Ollama + FAISS)")

with st.sidebar:
    st.header("⚙️ Setup")
    model = st.text_input("Ollama model", value="llama3.2",
                          help="Run `ollama pull llama3.2` first. Other options: qwen2.5, mistral, phi3")
    k = st.slider("Chunks retrieved (k)", 2, 10, 5)
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🚀 Process documents", type="primary", disabled=not files):
        with st.spinner("Reading, splitting and indexing..."):
            try:
                store, chunks, names = build_index(files)
                st.session_state.update(store=store, chunks=chunks, names=names)
                st.success(f"Indexed {len(names)} file(s), {len(chunks)} chunks")
            except Exception as e:
                st.error(str(e))
    if "names" in st.session_state:
        st.markdown("**Loaded:**")
        for n in st.session_state["names"]:
            st.markdown(f"- {n}")

if "store" not in st.session_state:
    st.info("👈 Upload one or more PDFs and click **Process documents** to begin.")
    st.stop()

store, chunks, names = (st.session_state[x] for x in ("store", "chunks", "names"))
llm = get_llm(model)

tab_qa, tab_sum, tab_cmp = st.tabs(["💬 Question Answering", "📝 Summarization", "🔀 Compare Documents"])

# ---- Question answering -----------------------------------------------------
with tab_qa:
    scope = st.selectbox("Search in", ["All documents"] + names, key="qa_scope")
    q = st.text_input("Ask a question", placeholder="e.g. What are the termination conditions?")
    if st.button("Ask", key="ask") and q:
        with st.spinner("Retrieving and thinking..."):
            try:
                kwargs = {"k": k}
                if scope != "All documents":
                    kwargs["filter"] = {"source": scope}
                docs = store.as_retriever(search_kwargs=kwargs).invoke(q)
                parser = PydanticOutputParser(pydantic_object=DocAnalysis)
                res = run_chain(QA_PROMPT, parser, llm, context=fmt(docs), question=q)
                show_analysis(res)
                with st.expander("Retrieved context"):
                    for d in docs:
                        st.markdown(f"**{d.metadata['source']} – page {d.metadata['page']}**")
                        st.caption(d.page_content)
            except Exception as e:
                st.error(f"Failed: {e}. Is Ollama running and the model pulled?")

# ---- Summarization ----------------------------------------------------------
with tab_sum:
    doc = st.selectbox("Document", names, key="sum_doc")
    if st.button("Summarize", key="sum"):
        with st.spinner("Analysing document..."):
            try:
                sel = sample_chunks(chunks, doc)
                parser = PydanticOutputParser(pydantic_object=DocAnalysis)
                res = run_chain(SUMMARY_PROMPT, parser, llm, name=doc, context=fmt(sel))
                show_analysis(res)
            except Exception as e:
                st.error(f"Failed: {e}")

# ---- Comparison -------------------------------------------------------------
with tab_cmp:
    if len(names) < 2:
        st.warning("Upload at least two PDFs to compare.")
    else:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Document A", names, index=0, key="cmp_a")
        b = c2.selectbox("Document B", names, index=1, key="cmp_b")
        focus = st.text_input("Focus (optional)", value="Overall differences, obligations and risks")
        if st.button("Compare", key="cmp"):
            if a == b:
                st.error("Choose two different documents.")
            else:
                with st.spinner("Comparing..."):
                    try:
                        ra = store.as_retriever(search_kwargs={"k": k, "filter": {"source": a}}).invoke(focus)
                        rb = store.as_retriever(search_kwargs={"k": k, "filter": {"source": b}}).invoke(focus)
                        parser = PydanticOutputParser(pydantic_object=Comparison)
                        res = run_chain(COMPARE_PROMPT, parser, llm, a=a, b=b,
                                        ctx_a=fmt(ra), ctx_b=fmt(rb), focus=focus)
                        show_comparison(res)
                    except Exception as e:
                        st.error(f"Failed: {e}")
