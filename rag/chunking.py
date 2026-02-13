from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def chunk_documents(docs: List[Document]) -> List[Document]:
    return splitter.split_documents(docs)
