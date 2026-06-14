from langchain_openai import OpenAIEmbeddings
from config import SILICONFLOW_API_KEY, EMBEDDING_API_URL, EMBEDDING_MODEL


def get_langchain_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=EMBEDDING_API_URL,
    )

