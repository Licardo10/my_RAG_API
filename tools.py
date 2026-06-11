import json

calculate_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行数学计算，支持加减乘除、幂运算等",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "descroption": "数学表达式，例如 '123 * 456' 或 '2**10'",
                }
            },
            "required": ["expression"],
        },
    },
}


def run_calculator(expression: str) -> str:
    allowed_chars = set("0123456789+-*/().%**")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含非法字符"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


