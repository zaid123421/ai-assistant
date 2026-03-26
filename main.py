import os
import shutil
from dotenv import load_dotenv
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


def ask(query, vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
    relevant = vector_store.similarity_search(query, k=12)
    context = "\n".join([d.page_content for d in relevant])
    prompt = f"""You are a helpful assistant for SweetSpot / company documents.
        RULES:
        1. For questions about content that appears in the CONTEXT below, answer ONLY using that context. Do not invent facts.
        2. For greetings or small talk (e.g. hi, hello, thanks), reply briefly and politely in one or two sentences, then invite the user to ask about the company or services.
        3. If the user asks something that is NOT answered by the context (or the context is empty for that topic), do NOT stay silent. Reply clearly that you do not have that information in the provided documents, and suggest they ask about topics that are in the documents (e.g. services, about us, contact).
        Context:
        {context}
        Question: {query}
        Answer:"""
    return llm.invoke(prompt)


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