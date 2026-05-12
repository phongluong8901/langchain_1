# /graph/nodes/grade_documents.py

from typing import Any, Dict

# Import 'máy chấm điểm' - dùng LLM để đánh giá độ liên quan của tài liệu
from graph.chains.retrieval_grader import retrieval_grader 
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Nhiệm vụ: Kiểm tra xem đống tài liệu vừa lấy từ DB (Retrieved) có thực sự trả lời được 
    cho câu hỏi không. Nếu có tài liệu nào 'lạc đề', hệ thống sẽ bật đèn tín hiệu đi Search Web.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---") # Log để theo dõi trạm đang chạy
    
    # 1. Lấy câu hỏi và danh sách tài liệu thô từ 'giỏ hàng' (State)
    question = state["question"]
    documents = state["documents"]

    # 2. Khởi tạo danh sách chứa tài liệu 'sạch' và biến đánh dấu (Flag) Search Web
    filtered_docs = [] # Chỉ chứa những tài liệu đạt chuẩn (Grade: Yes)
    web_search = False # Mặc định là False, nếu gặp rác sẽ bật lên True

    # 3. Vòng lặp kiểm tra từng tài liệu một (Giống như soi từng linh kiện trên băng chuyền)
    for d in documents:
        # Gọi máy chấm điểm: Đưa Câu hỏi + Nội dung tài liệu vào cho LLM thẩm định
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        
        # Lấy kết quả trả về (thường là 'yes' hoặc 'no')
        grade = score.binary_score
        
        if grade.lower() == "yes":
            # Nếu đạt chuẩn -> Cho vào danh sách giữ lại
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            # Nếu không liên quan -> Loại bỏ và kích hoạt tín hiệu đi Search Web bổ sung
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True 
            continue # Bỏ qua đoạn này, check tiếp đoạn sau
            
    # 4. Cập nhật lại giỏ hàng: 
    # Thay 'documents' cũ bằng 'filtered_docs' sạch hơn và gửi kèm tín hiệu 'web_search'
    return {"documents": filtered_docs, "question": question, "web_search": web_search}