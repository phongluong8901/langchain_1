# /graph/chains/hallucination_grader.py

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

# 1. Khởi tạo LLM (Vẫn là Qwen qua Ollama)
llm = ChatOllama(
    model="qwen3:1.7b", 
    temperature=0 # Rất quan trọng: Set bằng 0 để máy chấm điểm cực kỳ khắt khe và ổn định
)

# 2. Định nghĩa cấu trúc kết quả chấm điểm
class GradeHallucinations(BaseModel):
    """Điểm số nhị phân để kiểm tra xem câu trả lời có bị 'ảo giác' không."""

    # Kết quả trả về là một biến kiểu Boolean (True/False) 
    # tương ứng với việc câu trả lời có dựa trên sự thật (grounded) hay không.
    binary_score: bool = Field(
        description="Câu trả lời có dựa trên các sự thật (facts) hay không, 'yes' hoặc 'no'"
    )

# 3. Kỹ thuật Structured Output
# Ép LLM phải xuất ra dữ liệu đúng định dạng JSON để hệ thống tự động xử lý được
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 4. Thiết lập Chỉ dẫn kiểm định (System Prompt)
system = """Bạn là một giám khảo đánh giá liệu câu trả lời của LLM có dựa trên (grounded in) 
hoặc được hỗ trợ bởi một tập hợp các sự kiện (facts) được cung cấp hay không.
- Trả về 'yes' (True) nếu câu trả lời được hỗ trợ bởi dữ liệu.
- Trả về 'no' (False) nếu câu trả lời chứa thông tin không có trong dữ liệu cung cấp."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # Đối chiếu: 'documents' là sự thật, 'generation' là câu trả lời AI vừa viết
        ("human", "Tập hợp sự thật: \n\n {documents} \n\n Câu trả lời của LLM: {generation}"),
    ]
)

# 5. Tạo Chain kiểm định ảo giác
# Luồng: Lấy dữ liệu -> Chấm điểm -> Trả về kết quả True/False
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader