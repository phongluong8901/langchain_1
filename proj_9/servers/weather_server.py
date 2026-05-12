# weather_server.py

# Dòng 1: Import List để hỗ trợ định nghĩa kiểu dữ liệu (mặc dù trong code này chưa dùng tới List).
from typing import List

# Dòng 2: Import lớp FastMCP để khởi tạo server nhanh.
from mcp.server.fastmcp import FastMCP

# Dòng 3: Khởi tạo MCP Server với tên định danh là "Weather".
mcp = FastMCP("Weather")

# Dòng 4: Decorator đánh dấu hàm dưới đây là một công cụ (Tool) dành cho AI.
@mcp.tool()
async def get_weather(location: str) -> str:
    # Dòng 5: Từ khóa 'async def' cực kỳ quan trọng. 
    # Nó cho phép server xử lý tác vụ này mà không làm nghẽn các yêu cầu khác.
    # Trong thực tế, việc lấy thời tiết thường phải gọi API (như OpenWeatherMap), 
    # dùng 'async' giúp server rảnh tay làm việc khác trong khi chờ dữ liệu mạng trả về.

    # Dòng 6: Docstring mô tả chức năng để AI biết khi nào cần gọi hàm này.
    """Get weather for location."""
    
    # Dòng 7: Trả về một chuỗi văn bản. 
    # AI sẽ nhận được chuỗi này và dùng nó để trả lời người dùng.
    return "Hot as hell"

# Dòng 8: Điểm bắt đầu của chương trình.
if __name__ == "__main__":
    # Dòng 9: Lệnh chạy server với giao thức SSE (Server-Sent Events).
    # transport="sse": Thay vì chạy trong terminal như 'stdio', server này sẽ tạo ra 
    # một địa chỉ HTTP (thường là http://localhost:8000/sse).
    # AI Client sẽ kết nối tới địa chỉ này để trao đổi dữ liệu.
    mcp.run(transport="sse")