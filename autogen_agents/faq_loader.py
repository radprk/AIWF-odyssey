# faq_loader.py  
from pathlib import Path
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:  # pragma: no cover - fallback for older installs
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 
def load_faq_database(faq_path: str | Path):
    loader = TextLoader(str(faq_path), encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # pick any local HF model
        model_kwargs={"device": "cpu"}
    )
    return FAISS.from_documents(chunks, embeddings)
