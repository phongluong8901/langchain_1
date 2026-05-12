# /graph/nodes/retrieve.py

from typing import Any, Dict # Import công cụ định nghĩa kiểu dữ liệu (từ điển, bất kỳ)

from graph.state import GraphState # Import cấu trúc bộ nhớ chung của hệ thống (State)
from ingestion import retriever # Import "Cánh tay robot" dùng để lục tìm dữ liệu đã cài đặt ở file ingestion


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Nhiệm vụ: Dựa vào câu hỏi của người dùng, trạm này sẽ nhảy vào VectorDB (ChromaDB) 
    để "lôi" ra những đoạn văn bản có nội dung liên quan nhất.
    """

    print("---RETRIEVE---") # In ra Terminal để ông biết hệ thống đã bắt đầu lục kho dữ liệu
    
    # 1. Lấy câu hỏi từ trong "giỏ hàng" (State) ra
    # Đây là câu hỏi mà người dùng vừa nhập vào máy
    question = state["question"]

    # 2. Vận hành bộ máy tìm kiếm (Retriever)
    # Hàm .invoke(question) sẽ so sánh Vector của câu hỏi với Vector của tài liệu trong kho
    # Kết quả trả về là một danh sách các "Documents" (các đoạn văn bản tiềm năng)
    documents = retriever.invoke(question)

    # 3. Chuyển nguyên liệu về kho chứa chung
    # Trả về kết quả để cập nhật vào State, chuẩn bị cho trạm "Lọc" (Grade) xử lý tiếp
    return {"documents": documents, "question": question}