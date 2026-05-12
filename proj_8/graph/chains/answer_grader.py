# /graph/chains/answer_grader.py

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

# 1. Định nghĩa khuôn mẫu đánh giá
class GradeAnswer(BaseModel):
    """Điểm số nhị phân để xác nhận câu trả lời có giải quyết được câu hỏi không."""

    # Trả về True (Yes) nếu trả lời đúng ý, False (No) nếu trả lời lạc đề
    binary_score: bool = Field(
        description="Câu trả lời có giải quyết được câu hỏi hay không, 'yes' hoặc 'no'"
    )

# 2. Khởi tạo LLM
llm = ChatOllama(
    model="qwen3:1.7b", 
    temperature=0 # Luôn để 0 cho các tác vụ chấm điểm để đảm bảo tính khắt khe
)

# 3. Ép kiểu đầu ra (Structured Output)
# Bắt con Qwen phải trả về Object chứa biến binary_score kiểu Boolean
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 4. Thiết lập Chỉ dẫn (System Prompt) cho Giám khảo
system = """Bạn là một giám khảo đánh giá liệu một câu trả lời có giải quyết hoặc trả lời được 
đúng trọng tâm câu hỏi hay không.
- Trả về 'yes' (True) nếu câu trả lời giải quyết được câu hỏi.
- Trả về 'no' (False) nếu câu trả lời không trực tiếp trả lời hoặc bỏ sót ý của câu hỏi."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # Đối chiếu trực tiếp câu hỏi gốc và câu trả lời AI vừa viết
        ("human", "Câu hỏi người dùng: \n\n {question} \n\n Câu trả lời của LLM: {generation}"),
    ]
)

# 5. Tạo Chain kiểm định cuối cùng
# Luồng: Nhận dữ liệu -> Chấm điểm độ liên quan -> Trả về True/False
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader