from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from embedding import get_langchain_embeddings
from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
)


def build_vectorstore():
    embeddings = get_langchain_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def build_LLM():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )


def load_document(file_path: str, vectorstore: Chroma):
    with open(file_path, "r", encoding="UTF-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "?", "!", " "],
        length_function=len,
    )

    chunks = splitter.split_text(raw_text)

    print(f"原始文本长度: {len(raw_text)} 字符")
    print(f"分块数量: {len(chunks)}")

    vectorstore.add_texts(texts=chunks)
    print(f"成功存入 {vectorstore._collection.count()} 个向量")


def init_RAG_engine(file_path: str):
    vectorstore = build_vectorstore()
    count = vectorstore._collection.count()

    if count == 0:
        load_document(file_path, vectorstore)
    else:
        print(f"向量库已存在，共 {count} 个向量")

    LLM = build_LLM()

    def query_RAG(question: str) -> str:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        system_prompt = f"""
        你是一个专业的技术助手。请基于以下【参考资料】回答用户的问题。
        如果参考资料中没有相关信息，请如实说“资料中没有提及”，不要编造。

        【参考资料】
        {context}

        【用户问题】
        {question}

        【回答】"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        response = LLM.invoke(messages)

        return response.content

    return query_RAG
