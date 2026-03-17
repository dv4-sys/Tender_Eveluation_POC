# vector_store.py
import faiss
import numpy as np
from typing import List
from langchain_core.documents import Document
from bedrock_embedding import BedrockEmbedding

class VectorStore:
    def __init__(self):
        self.embedding_model = BedrockEmbedding()
        self.index = None
        self.documents = []

    def build_index(self, docs: List[Document]):
        embeddings = []

        for doc in docs:
            emb = self.embedding_model.embed(doc.page_content[:8000])
            embeddings.append(emb)
            self.documents.append(doc)

        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def search(self, query: str, k: int = 5):
        q_emb = self.embedding_model.embed(query)
        D, I = self.index.search(np.array([q_emb]), k)

        return [self.documents[i] for i in I[0]]