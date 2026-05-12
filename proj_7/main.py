from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

# Import các thành phần đã định nghĩa ở các file trước
from chains import revisor, first_responder
from tool_executor import execute_tools

# 1. Cấu hình giới hạn: Agent chỉ được phép đi tìm kiếm tối đa 2 lần để tránh tốn kém và lặp vô tận
MAX_ITERATIONS = 2

# 2. Định nghĩa các Node (Các bước xử lý)
def draft_node(state: MessagesState):
    """Bước 1: Tạo bản thảo đầu tiên."""
    # Gọi first_responder để viết câu trả lời sơ khai và đề xuất từ khóa tìm kiếm
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Bước 2: Sửa đổi câu trả lời dựa trên kết quả tìm được từ Internet."""
    # Gọi revisor để đọc kết quả từ Tool và viết lại bài có trích dẫn nguồn
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# 3. Hàm điều phối vòng lặp (Event Loop)
def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Kiểm tra xem Agent đã đi tìm kiếm đủ số lần chưa."""
    # Đếm số lượng ToolMessage trong lịch sử để biết Agent đã chạy Tool mấy lần
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits
    
    # Nếu vượt quá giới hạn cho phép -> Kết thúc (END)
    if num_iterations > MAX_ITERATIONS:
        return END
    # Nếu chưa đủ -> Tiếp tục đi tìm kiếm thông tin (execute_tools)
    return "execute_tools"

# 4. Xây dựng Sơ đồ luồng (Graph)
builder = StateGraph(MessagesState)

# Thêm các nút chức năng
builder.add_node("draft", draft_node)           # Nút viết nháp
builder.add_node("execute_tools", execute_tools) # Nút chạy Tavily Search
builder.add_node("revise", revise_node)         # Nút sửa bài

# Thiết lập các đường nối (Edges)
builder.add_edge(START, "draft")                 # Bắt đầu -> Viết nháp
builder.add_edge("draft", "execute_tools")       # Viết nháp xong -> Đi tìm kiếm luôn
builder.add_edge("execute_tools", "revise")      # Tìm kiếm xong -> Đưa dữ liệu cho bước sửa bài

# Thiết lập nhánh điều kiện sau khi sửa bài
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

# Biên dịch sơ đồ
graph = builder.compile()

# 5. Thực thi và lấy kết quả
if __name__ == "__main__":
    res = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
                }
            ]
        }
    )

    # Trích xuất câu trả lời cuối cùng từ cấu trúc Tool Call
    # Vì chúng ta ép AI trả về theo Schema, nên nội dung nằm trong args["answer"]
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("\n--- FINAL ANSWER ---\n")
        print(last_message.tool_calls[0]["args"]["answer"])