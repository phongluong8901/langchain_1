# /graph/nodes/generate.py

# Import các kiểu dữ liệu để khai báo rõ ràng (giúp code sạch và dễ debug)
from typing import Any, Dict 

# Import cái 'lõi' xử lý ngôn ngữ đã dựng sẵn (Prompt + LLM)
from graph.chains.generation import generation_chain 

# Import cấu trúc 'giỏ hàng' (State) để biết hệ thống đang lưu giữ những gì
from graph.state import GraphState 


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Hàm này nhận vào trạng thái hiện tại (State) và trả về nội dung AI vừa viết.
    """
    print("---GENERATE---") # In ra để ông theo dõi luồng chạy trên Terminal
    
    # Bước 1: Lấy 'nguyên liệu' từ trong giỏ hàng (State) ra
    question = state["question"]   # Câu hỏi ban đầu của người dùng
    documents = state["documents"] # Danh sách tài liệu đã được lọc (Grade) ở bước trước đó

    # Bước 2: Đưa nguyên liệu vào 'máy sản xuất' (Generation Chain)
    # Nó sẽ kết hợp Context (tài liệu) và Question (câu hỏi) để tạo ra văn bản trả lời
    generation = generation_chain.invoke({"context": documents, "question": question})

    # Bước 3: Cập nhật kết quả vào giỏ hàng
    # Trả về một Dictionary để LangGraph tự động ghi đè/bổ sung vào State chung của hệ thống
    return {"documents": documents, "question": question, "generation": generation}