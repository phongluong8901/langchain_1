# math_server.py

# Dòng 1: Import lớp FastMCP từ thư viện mcp. 
# FastMCP là một "framework" giúp việc tạo Server cực kỳ nhanh và đơn giản (tương tự như FastAPI).
from mcp.server.fastmcp import FastMCP

# Dòng 2: Khởi tạo một đối tượng MCP Server với tên là "Math".
# Tên này sẽ hiển thị trong log hoặc trong giao diện của AI Client để bạn biết server nào đang chạy.
mcp = FastMCP("Math")

# Dòng 3: Một "Decorator" dùng để đánh dấu hàm bên dưới là một "Tool".
# Khi có dòng này, AI mới có quyền nhìn thấy và sử dụng hàm `add`.
@mcp.tool()
def add(a: int, b: int) -> int:
    # Dòng 4: Docstring (Cực kỳ quan trọng!). 
    # AI sẽ đọc dòng này để hiểu: "À, khi người dùng muốn cộng số, mình phải gọi hàm này".
    """Add two numbers"""
    
    # Dòng 5: Logic Python bình thường, trả về tổng của a và b.
    return a + b

# Dòng 6: Tiếp tục khai báo một Tool khác cho server.
@mcp.tool()
def multiply(a: int, b: int) -> int:
    # Dòng 7: Mô tả cho AI biết đây là hàm dùng để nhân hai số.
    """Multiply two numbers"""
    
    # Dòng 8: Trả về tích của a và b.
    return a * b

# Dòng 9: Kiểm tra xem file này có đang được chạy trực tiếp hay không (không phải bị import).
if __name__ == "__main__":
    # Dòng 10: Lệnh quan trọng nhất để kích hoạt server.
    # transport="stdio": Nghĩa là server sẽ giao tiếp với AI qua luồng chuẩn Input/Output của Terminal.
    # Khi bạn chạy dòng này, server sẽ đứng đợi lệnh từ AI Client gửi đến.
    mcp.run(transport="stdio")