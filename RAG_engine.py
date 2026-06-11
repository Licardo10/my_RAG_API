from ChromaDB import get_embeddings, init_collection, add_documents
from llm_client import get_LLM_response
from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            for sep in ["。", "？", "！", "\n", ".", "?", "!"]:
                last_sep = text.rfind(sep, start, end)
                if (last_sep) > start + chunk_size // 2:
                    end = last_sep + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < text_len else end
        if start < 0:
            start = end
    return chunks


class RAGEngine:
    def __init__(self, collection=None):
        self.collection = collection if collection else init_collection()
        self.is_initialized = False

    def init_from_text_file(self, file_path: str):
        if self.collection.count() > 0:
            print(f"向量库已存在，共 {self.collection.count()} 个向量")
            return

        with open(file_path, "r", encoding="UTF-8") as f:
            raw_text = f.read()

        chunks = chunk_text(raw_text)

        print(f"原始文本长度: {len(raw_text)} 字符")
        print(f"分块数量: {len(chunks)}")

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": file_path, "chunk_index": i} for i in range(len(chunks))
        ]

        add_documents(self.collection, chunks, ids, metadatas)
        print(f"成功存入 {self.collection.count()} 个向量")

    def retrieve(self, query: str, n_results: int = 3):
        """检索最相关的文本块"""
        results = self.collection.query(
            query_embeddings=[get_embeddings(query)[0]],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        return results

    def generate_answer(self, query: str, context_chunks: list) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"""
        你是一个专业的技术助手。请基于以下【参考资料】回答用户的问题。
        如果参考资料中没有相关信息，请如实说“资料中没有提及”，不要编造。

        【参考资料】
        {context}

        【用户问题】
        {query}

        【回答】"""
        return get_LLM_response(prompt, temperature=0.3)

    def query(self, query: str) -> str:
        results = self.collection.query(
            query_embeddings=[get_embeddings(query)[0]],
            n_results=3,
            include=["documents"],
        )

        chunks = results["documents"][0]
        if not chunks:
            return "没有找到相关信息"

        answer = self.generate_answer(query, chunks)
        return answer
