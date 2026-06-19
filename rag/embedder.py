import os
from typing import Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
RESUME_TEXT_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index", "resume.txt")


def _embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


def create_vectorstore(text: str, api_key: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, _embeddings(api_key))


def save_vectorstore(vectorstore: FAISS, resume_text: str) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    with open(RESUME_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(resume_text)


def load_vectorstore(api_key: str) -> Optional[Tuple[FAISS, str]]:
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return None
    vectorstore = FAISS.load_local(
        INDEX_DIR, _embeddings(api_key), allow_dangerous_deserialization=True
    )
    resume_text = ""
    if os.path.exists(RESUME_TEXT_PATH):
        with open(RESUME_TEXT_PATH, "r", encoding="utf-8") as f:
            resume_text = f.read()
    return vectorstore, resume_text


def get_relevant_context(vectorstore: FAISS, query: str, k: int = 5) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)
