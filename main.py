import json
import os
import re
import shutil
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"


def message_to_text(msg) -> str:
    """Normalize AIMessage.content (str, or list of Gemini content blocks) to plain text."""
    if getattr(msg, "text", None):
        t = msg.text
        if isinstance(t, str) and t.strip():
            return t
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(c)


def load_documents(folder="documents"):
    """Load PDF, TXT, and Markdown from the documents folder"""
    docs = []
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found!")
        return docs
    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt") or file.endswith(".md"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        length_function=len,
    )
    return splitter.split_documents(docs)


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)


def _parse_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _search_web_snippets(user_query: str, max_chars: int = 8000) -> str:
    from langchain_community.tools import DuckDuckGoSearchRun

    tool = DuckDuckGoSearchRun()
    out = tool.run(user_query)
    if isinstance(out, str) and len(out) > max_chars:
        return out[:max_chars].rsplit("\n", 1)[0] + "\n…"
    return out or ""


def ask(query, vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
    route_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
    )
    relevant = vector_store.similarity_search(query, k=12)
    context = "\n".join([d.page_content for d in relevant])

    route_prompt = f"""You route questions for an assistant that has internal CONTEXT (company documents).

CONTEXT (may be partial):
---
{context}
---

USER QUESTION:
{query}

Reply with ONLY a JSON object (no markdown fences), exactly this shape:
{{"is_smalltalk": <true only for greetings, thanks, or brief chat that needs no facts>,
 "answered_from_docs": <true if CONTEXT fully supports a factual answer to the question>,
 "answer": "<when is_smalltalk or answered_from_docs is true, your reply; use CONTEXT only for facts; otherwise empty string>"}}

Rules:
- If is_smalltalk is true, set answered_from_docs to true and put a short polite reply in answer (invite questions about the company if appropriate).
- If the question asks for facts not in CONTEXT, set answered_from_docs to false, is_smalltalk to false, answer "".
- Never invent company facts; only use CONTEXT when answered_from_docs is true.
- In answer, never tell the user whether information came from documents, files, search, or the web—reply naturally."""

    route_raw = message_to_text(route_llm.invoke(route_prompt))
    data = _parse_json_object(route_raw)

    if data is None:
        prompt = f"""You are a helpful assistant.
RULES:
1. When CONTEXT contains the answer, use only that information for facts. Do not invent facts.
2. For greetings or small talk, reply briefly and politely.
3. If CONTEXT does not answer the question, reply briefly that you cannot answer it; do not mention files, documents, or search.

CONTEXT:
{context}
Question: {query}
Answer:"""
        return llm.invoke(prompt)

    is_smalltalk = data.get("is_smalltalk") is True
    answered_from_docs = data.get("answered_from_docs") is True
    routed_answer = (data.get("answer") or "").strip()

    if is_smalltalk or answered_from_docs:
        if not routed_answer:
            routed_answer = (
                "I don’t have enough to answer that; please try rephrasing your question."
            )
        return AIMessage(content=routed_answer)

    try:
        web_text = _search_web_snippets(query)
    except Exception as exc:
        return AIMessage(
            content=(
                f"I couldn’t get an answer right now ({exc}). "
                "Check your connection and try again later."
            )
        )

    if not web_text.strip():
        return AIMessage(
            content=(
                "I couldn’t find a useful answer. Try rephrasing the question."
            )
        )

    synth = f"""Reply in the SAME language as the user’s question (Arabic or English, etc.).

Use ONLY the facts supported by the REFERENCE TEXT below; if it is irrelevant or insufficient, say briefly that you cannot give a reliable answer. Do not tell the user where the information came from (no mention of websites, search, files, or documents).

USER QUESTION:
{query}

REFERENCE TEXT:
{web_text}

Your answer:"""
    return llm.invoke(synth)


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print("Files in documents:", os.listdir("documents"))

    if not docs:
        print("No documents found in the 'documents' folder!")
        exit()

    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Removed old chroma_db for a clean index.")

    print(f"Loaded {len(docs)} document(s). Splitting into chunks...")
    chunks = split_docs(docs)
    print(f"Total chunks: {len(chunks)}")

    print("Creating vector database...")
    vs = create_vector_store(chunks)

    test_hits = vs.similarity_search("SweetSpot company services", k=3)
    print("\n[Sanity check] First retrieved snippet:")
    print((test_hits[0].page_content[:500] + "...") if test_hits else "(no hits)")

    print("\n--- Ready! Type your question (or 'exit' to quit) ---\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            break
        answer = ask(q, vs)
        print(f"\nAnswer: {message_to_text(answer)}\n")