from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def preview_chunks(text: str, chunk_size: int, overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    return [
        {"index": i + 1, "content": c, "length": len(c)}
        for i, c in enumerate(chunks)
    ]


def search_with_scores(vectorstore: FAISS, query: str, k: int) -> list[dict]:
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "rank": i + 1,
            "content": doc.page_content,
            "score": float(score),  # FAISS는 L2 거리 (낮을수록 유사)
            "length": len(doc.page_content),
        }
        for i, (doc, score) in enumerate(results)
    ]
