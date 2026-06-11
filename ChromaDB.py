import chromadb
import requests
from config import (
    SILICONFLOW_API_KEY,
    EMBEDDING_API_URL,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
)


def get_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts if isinstance(texts, list) else [texts],
        "encoding_format": "float",
    }

    resp = requests.post(
        EMBEDDING_API_URL,
        json=payload,
        headers=headers,
        timeout=30,
    )

    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


def init_collection():
    chromadb_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chromadb_client.get_or_create_collection(name="rag_docs")
    return collection


def add_documents(collection, documents, ids, metadatas):
    embeddings = get_embeddings(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
