# 1. Khai báo các kiểu dữ liệu và thư viện hỗ trợ
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv() # Tải API Key từ file .env

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Import 2 Chain (Viết và Chê) mà bạn đã định nghĩa ở file chains.py trước đó
from chains import generate_chain, reflect_chain

# 2. Định nghĩa cấu trúc dữ liệu lưu trữ trạng thái của Graph
class MessageGraph(TypedDict):
    # Annotated + add_messages giúp cộng dồn (append) tin nhắn mới vào danh sách cũ 
    # thay vì ghi đè (overwrite) hoàn toàn mỗi khi node chạy.
    messages: Annotated[list[BaseMessage], add_messages]

# Hằng số định danh cho các Node
REFLECT = "reflect"
GENERATE = "generate"

# 3. Định nghĩa các Node (Các bước xử lý trong Graph)

def generation_node(state: MessageGraph):
    """Node này đóng vai trò 'Người viết'."""
    # Lấy toàn bộ tin nhắn hiện có gửi cho generate_chain và trả về tin nhắn mới của AI
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    """Node này đóng vai trò 'Người phê bình'."""
    # Gửi lịch sử cho reflect_chain để lấy lời chê
    res = reflect_chain.invoke({"messages": state["messages"]})
    # Chuyển nội dung lời chê thành HumanMessage để đánh lừa Agent ở bước sau 
    # rằng đây là ý kiến phản hồi từ người dùng/giám khảo.
    return {"messages": [HumanMessage(content=res.content)]}

# 4. Xây dựng Sơ đồ (Graph)
builder = StateGraph(state_schema=MessageGraph)

# Thêm các Node vào sơ đồ
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

# Điểm bắt đầu luôn là viết bài (GENERATE)
builder.set_entry_point(GENERATE)

# 5. Logic điều hướng (Conditional Edges)
def should_continue(state: MessageGraph):
    """Hàm kiểm tra xem đã đến lúc dừng lại chưa."""
    # Nếu danh sách tin nhắn > 6 (tương đương khoảng 3 vòng lặp Viết-Chê-Viết)
    if len(state["messages"]) > 6:
        return END # Dừng lại và trả kết quả cuối cùng
    return REFLECT # Nếu chưa đủ số lần, đẩy sang cho ông 'Chê'

# Sau khi GENERATE xong, chạy hàm should_continue để quyết định đi tiếp hay dừng
builder.add_conditional_edges(GENERATE, should_continue)

# Sau khi REFLECT xong, bắt buộc quay lại GENERATE để sửa bài
builder.add_edge(REFLECT, GENERATE)

# 6. Biên dịch và Hiển thị sơ đồ
graph = builder.compile()
# In ra mã Mermaid (để vẽ biểu đồ) và sơ đồ dạng chữ (ASCII) để kiểm tra luồng
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

# 7. Thực thi chương trình
if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:
                            @LangChainAI
                — newly Tool Calling feature is seriously underrated.
                ... (nội dung tweet gốc) ...
                """
            )
        ]
    }
    # Chạy vòng lặp tự sửa lỗi
    response = graph.invoke(inputs)
    # In ra toàn bộ quá trình hội thoại và kết quả cuối cùng
    print(response)