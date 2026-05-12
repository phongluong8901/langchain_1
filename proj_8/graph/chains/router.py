# /graph/chains/router.py

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field # Công cụ để ép LLM trả về dữ liệu đúng cấu trúc
from langchain_ollama import ChatOllama

# 1. Định nghĩa cấu trúc đầu ra (Schema) cho Router
class RouteQuery(BaseModel):
    """
    Sử dụng Pydantic để bắt LLM phải trả về đúng định dạng JSON 
    có trường 'datasource' với 1 trong 2 giá trị cố định.
    """
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Dựa trên câu hỏi của người dùng, chọn hướng đi tới web search hoặc vectorstore.",
    )

# 2. Khởi tạo LLM (Ollama với Qwen)
llm = ChatOllama(
    model="qwen3:1.7b", 
    temperature=0 # Set bằng 0 để kết quả phân loại luôn nhất quán, không "sáng tạo"
)

# 3. Kỹ thuật Structured Output
# Ép con Qwen phải trả về kết quả theo đúng khuôn mẫu của class RouteQuery đã định nghĩa ở trên
structured_llm_router = llm.with_structured_output(RouteQuery)

# 4. Thiết lập Chỉ dẫn (System Prompt) cho Router
system = """Bạn là một chuyên gia điều phối câu hỏi.
- Kho dữ liệu (vectorstore) của bạn chứa các tài liệu về: AI Agents, Prompt Engineering, và Adversarial Attacks (tấn công đối kháng).
- Sử dụng 'vectorstore' cho các câu hỏi liên quan đến các chủ đề trên.
- Với tất cả các chủ đề khác (tin tức, thời tiết, đời sống...), hãy sử dụng 'websearch'."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 5. Tạo Chain điều hướng
# Luồng đi: Lấy Prompt -> Đưa vào LLM -> LLM xuất ra JSON theo kiểu RouteQuery
question_router = route_prompt | structured_llm_router