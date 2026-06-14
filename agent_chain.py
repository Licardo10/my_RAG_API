from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL


@tool
def calculator(expression: str) -> str:
    """执行数学计算，支持加减乘除、幂运算等

    Args:
        expression: 数学表达式，例如 '123 * 456' 或 '2**10'
    """
    allowed_chars = set("0123456789+-*/().%**")
    if not all(c in allowed_chars for c in expression):
        return "ERROR"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"ERROR {e}"


def build_agent():
    LLM = ChatOpenAI(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
    )

    agent = create_agent(
        LLM,
        tools=[calculator],
        system_prompt="你是一个智能助手，可以使用工具来回答问题",
    )

    def agent_invoke(question: str) -> str:
        result = agent.invoke({"message": [HumanMessage(content=question)]})
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                return msg.content
        return ""

    return agent_invoke
