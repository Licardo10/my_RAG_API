import os
from dotenv import load_dotenv

load_dotenv()

# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# Embedding
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBEDDING_API_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-m3"

# vector database
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "RAG_docs"

# chunk
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# data source
DOCUMENT_PATH = "./data/tech_article.txt"
