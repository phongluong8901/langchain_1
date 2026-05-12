# 1. Khai báo các thư viện cần thiết
from dotenv import load_dotenv  # Thư viện để đọc các secret key (API Key) từ file .env

from langchain_core.messages import HumanMessage  # Class đại diện cho tin nhắn từ phía người dùng
from langgraph.graph import MessagesState, StateGraph, END  # Các thành phần để xây dựng cấu trúc sơ đồ (Graph)

# Import 2 hàm quan trọng từ file nodes.py (bạn cần có file này để code chạy được)
from nodes import run_agent_reasoning, tool_node 

# 2. Tải các biến môi trường (như GOOGLE_API_KEY, TAVILY_API_KEY)
load_dotenv() 

# 3. Định nghĩa các hằng số để quản lý tên các bước (Nodes) trong sơ đồ
AGENT_REASON = "agent_reason"  # Tên bước: AI suy nghĩ và quyết định hành động
ACT = "act"                    # Tên bước: Thực thi các công cụ (Tools)
LAST = -1                      # Chỉ số để lấy tin nhắn cuối cùng trong danh sách messages

# 4. Hàm điều kiện để quyết định luồng đi tiếp hay dừng lại
def should_continue(state: MessagesState) -> str:
    """
    Hàm này kiểm tra tin nhắn cuối cùng của AI:
    - Nếu AI yêu cầu gọi Tool (tool_calls) -> Chuyển sang bước ACT.
    - Nếu AI trả lời trực tiếp (không dùng tool) -> Kết thúc (END).
    """
    if not state["messages"][LAST].tool_calls:
        return END  # Dừng chương trình và trả kết quả cho người dùng
    return ACT      # Tiếp tục sang bước thực hiện công cụ

# 5. Khởi tạo và xây dựng Sơ đồ luồng (Graph)
# MessagesState giúp lưu lại toàn bộ lịch sử hội thoại giữa các bước
flow = StateGraph(MessagesState)

# Thêm bước Suy nghĩ vào sơ đồ
flow.add_node(AGENT_REASON, run_agent_reasoning)
# Quy định đây là điểm bắt đầu khi chạy ứng dụng
flow.set_entry_point(AGENT_REASON)

# Thêm bước Hành động (thực thi tool) vào sơ đồ
flow.add_node(ACT, tool_node)

# Thiết lập nhánh điều hướng sau khi Agent suy nghĩ xong
flow.add_conditional_edges(
    AGENT_REASON,     # Sau bước suy nghĩ...
    should_continue,  # ...chạy hàm này để kiểm tra...
    {
        END: END,     # ...nếu hàm trả về END thì dừng.
        ACT: ACT      # ...nếu hàm trả về ACT thì nhảy sang node ACT.
    }
)

# Sau khi thực hiện công cụ ở bước ACT, luồng PHẢI quay lại bước Suy nghĩ
# để AI đọc kết quả từ công cụ và quyết định làm gì tiếp theo.
flow.add_edge(ACT, AGENT_REASON)

# 6. Biên dịch sơ đồ thành ứng dụng thực thi (App)
app = flow.compile()

# Xuất sơ đồ ra file ảnh flow.png để hình dung luồng làm việc
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

# 7. Chạy chương trình chính
if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    
    # Gửi yêu cầu đầu tiên vào hệ thống
    # Yêu cầu này gồm 2 bước: Tìm nhiệt độ (Search) và Nhân ba (Triple)
    res = app.invoke({
        "messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]
    })
    
    # In ra nội dung tin nhắn cuối cùng (Sau khi AI đã tổng hợp mọi thông tin)
    print(res["messages"][LAST].content)