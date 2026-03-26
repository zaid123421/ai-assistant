import os
import shutil

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SweetSpot RAG Q&A", page_icon="💬")

# Cloud: Streamlit Secrets | Local: .env (after set_page_config)
if os.environ.get("GOOGLE_API_KEY") in (None, ""):
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

from main import CHROMA_DIR, ask, create_vector_store, load_documents, split_docs


@st.cache_resource(show_spinner="Building knowledge index...")
def get_vector_store():
    docs = load_documents()
    if not docs:
        return None
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    chunks = split_docs(docs)
    return create_vector_store(chunks)


st.title("SweetSpot — Ask the documents")

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("Missing `GOOGLE_API_KEY`. Add it to Streamlit Secrets or `.env`.")
    st.stop()

vs = get_vector_store()
if vs is None:
    st.error("No documents found in the `documents/` folder.")
    st.stop()

question = st.text_input("Your question", placeholder="e.g. What services does SweetSpot offer?")
if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Answering..."):
        response = ask(question.strip(), vs)
    st.markdown(response.content)