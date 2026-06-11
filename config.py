import os
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"
LLM_MODEL = "deepseek-chat"
CHROMA_PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DOCUMENT_PATH = "./data/tech_article.txt"