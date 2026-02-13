import hashlib
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .config import embeddings, VECTORSTORE_ROOT

def hash_docs(docs: List[Document]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.page_content.encode("utf-8"))
    return h.hexdigest()

def load_or_create_store(docs: List[Document], store_id: str) -> FAISS:
    path = VECTORSTORE_ROOT / store_id

    if path.exists():
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    store = FAISS.from_documents(docs, embeddings)
    store.save_local(path)
    return store
