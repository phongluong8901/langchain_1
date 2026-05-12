from dotenv import load_dotenv

load_dotenv() # Tải API Key (TAVILY_API_KEY) từ file .env

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

# Import các Schema để lấy tên class làm tên công cụ
from schemas import AnswerQuestion, ReviseAnswer

# 1. Khởi tạo công cụ tìm kiếm Tavily, giới hạn lấy 5 kết quả mỗi lần tìm
tavily_tool = TavilySearch(max_results=5)

# 2. Định nghĩa hàm xử lý việc tìm kiếm hàng loạt
def run_queries(search_queries: list[str], **kwargs):
    """Hàm này nhận vào danh sách các từ khóa và chạy tìm kiếm đồng thời."""
    # .batch giúp chạy nhiều query cùng lúc thay vì chạy từng cái một, giúp tăng tốc độ
    return tavily_tool.batch([{"query": query} for query in search_queries])

# 3. Khởi tạo ToolNode - "Nút thực thi" trong sơ đồ LangGraph
execute_tools = ToolNode(
    [
        # Biến hàm run_queries thành một công cụ có cấu trúc (StructuredTool)
        # Việc gán name=AnswerQuestion.__name__ rất quan trọng:
        # Nó giúp LangGraph hiểu rằng khi AI gọi schema "AnswerQuestion", 
        # thì thực tế là nó đang muốn chạy hàm tìm kiếm này.
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)