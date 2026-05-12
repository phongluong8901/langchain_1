# /graph/chains/retrieve_grader.py

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field # Công cụ định nghĩa khuôn mẫu dữ liệu
from langchain_ollama import ChatOllama

# 1. Khởi tạo LLM (Sử dụng Qwen qua Ollama)
llm = ChatOllama(
    model="qwen3:1.7b", 
    temperature=0 # Giữ nhiệt độ bằng 0 để kết quả chấm điểm khách quan và nhất quán
)

# 2. Định nghĩa "Tiêu chuẩn kỹ thuật" cho kết quả chấm điểm
class GradeDocuments(BaseModel):
    """Điểm số nhị phân (Yes/No) để kiểm tra độ liên quan của tài liệu."""

    # Ép LLM phải trả về một biến 'binary_score' kiểu chuỗi
    binary_score: str = Field(
        description="Tài liệu có liên quan đến câu hỏi hay không, trả về 'yes' hoặc 'no'"
    )

# 3. Kỹ thuật Structured Output
# Buộc con LLM phải đóng vai một cái máy chấm điểm, đầu ra chỉ được phép là JSON theo Class GradeDocuments
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 4. Thiết lập "Quy trình kiểm định" (System Prompt)
system = """Bạn là một giám khảo đánh giá mức độ liên quan của tài liệu được trích xuất đối với câu hỏi của người dùng.
- Nếu tài liệu chứa từ khóa hoặc có ý nghĩa ngữ nghĩa liên quan đến câu hỏi, hãy chấm là 'yes'.
- Trả về kết quả nhị phân 'yes' hoặc 'no' để cho biết tài liệu có hữu ích hay không."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # Truyền cả tài liệu thô và câu hỏi vào để LLM đối chiếu
        ("human", "Tài liệu trích xuất: \n\n {document} \n\n Câu hỏi người dùng: {question}"),
    ]
)

# 5. Tạo Chain chấm điểm
# Luồng: Ghép Prompt -> Đưa qua LLM -> Trả về kết quả 'yes' hoặc 'no' đã được định dạng
retrieval_grader = grade_prompt | structured_llm_grader