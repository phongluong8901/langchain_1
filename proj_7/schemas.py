from typing import List
from pydantic import BaseModel, Field

# 1. Định nghĩa cấu trúc của phần "Phê bình" (Reflection)
class Reflection(BaseModel):
    # 'missing': AI phải tự chỉ ra những thông tin quan trọng nào còn thiếu
    missing: str = Field(description="Critique of what is missing.")
    # 'superfluous': AI phải tự chỉ ra những phần nào đang bị thừa thãi hoặc lan man
    superfluous: str = Field(description="Critique of what is superfluous")

# 2. Định nghĩa cấu trúc cho câu trả lời đầu tiên (AnswerQuestion)
class AnswerQuestion(BaseModel):
    """Answer the question.""" # Docstring này giúp AI hiểu mục đích của class này

    # 'answer': Nội dung câu trả lời chính, yêu cầu chi tiết khoảng 250 từ
    answer: str = Field(description="~250 word detailed answer to the question.")
    
    # 'reflection': Lồng class Reflection ở trên vào đây để AI tự đánh giá bài viết của mình
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    
    # 'search_queries': Danh sách 1-3 từ khóa tìm kiếm để AI dùng đi tra cứu bổ sung thông tin
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

# 3. Định nghĩa cấu trúc cho câu trả lời đã được sửa đổi (ReviseAnswer)
# Class này kế thừa (Inheritance) từ AnswerQuestion, nghĩa là nó có sẵn tất cả các trường ở trên
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    # Thêm trường 'references': Danh sách các link nguồn hoặc tài liệu trích dẫn
    # Đây là phần cực kỳ quan trọng để đảm bảo tính xác thực của thông tin
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )