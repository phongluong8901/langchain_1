import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

# 1. Tải các biến môi trường từ file .env (chứa API Key)
load_dotenv()

# --- CẤU HÌNH ---
# Số lần tối đa Agent được phép suy nghĩ/gọi công cụ để tránh vòng lặp vô tận
MAX_ITERATIONS = 10
# Tên model Gemini 3.1 bạn đang sử dụng
MODEL = "models/gemini-3.1-flash-lite" 

# --- CÔNG CỤ (TOOLS) ---
# Dùng decorator @tool để biến một hàm Python thành một công cụ mà AI có thể hiểu được

@tool
def get_product_price(product: str) -> float:
    """Tra cứu giá của một sản phẩm trong danh mục."""
    # Dòng print này giúp bạn theo dõi khi nào AI thực sự gọi hàm này
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)   #1299.99

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Áp dụng mức giảm giá vào giá gốc và trả về giá cuối cùng.
    Các mức giảm giá có sẵn: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    # Tính toán và làm tròn kết quả
    return round(price * (1 - discount / 100), 2) #$1000.99

# --- VÒNG LẶP CỦA AGENT (AGENT LOOP) ---
# @traceable giúp gửi dữ liệu lên LangSmith để bạn theo dõi luồng chạy (nếu có cấu hình)
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    # Khai báo danh sách các công cụ cho Agent
    tools = [get_product_price, apply_discount]
    # Tạo một từ điển để ánh xạ tên tool với hàm tương ứng nhằm gọi hàm nhanh hơn
    tools_dict = {t.name: t for t in tools}

    # Khởi tạo mô hình AI. model_provider="google_genai" là bắt buộc để dùng Gemini API
    llm = init_chat_model(MODEL, model_provider="google_genai", temperature=0)
    
    # "Trang bị" các công cụ cho AI, giúp AI biết mình có quyền gọi những hàm nào
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # Khởi tạo lịch sử hội thoại
    messages = [
        # SystemMessage thiết lập "luật chơi" cứng rắn cho AI
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES:\n"
                "1. NEVER guess product prices. Call get_product_price.\n"
                "2. Call apply_discount ONLY after getting price from tool.\n"
                "3. NEVER calculate math yourself.\n"
                "4. Ask for discount tier if not specified."
            )
        ),
        # HumanMessage chứa câu hỏi thực tế của bạn
        HumanMessage(content=question),
    ]

    # Bắt đầu vòng lặp suy nghĩ
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Gửi toàn bộ lịch sử tin nhắn cho AI để nhận phản hồi (Action)
        ai_message = llm_with_tools.invoke(messages)
        
        # Kiểm tra xem AI muốn trả lời người dùng hay muốn gọi Tool
        tool_calls = ai_message.tool_calls

        # Nếu không có yêu cầu gọi Tool nào, AI đã có câu trả lời cuối cùng
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Nếu AI yêu cầu gọi Tool, lấy thông tin tool đầu tiên trong danh sách
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")      # Tên hàm (vd: get_product_price)
        tool_args = tool_call.get("args", {})  # Tham số (vd: product='laptop')
        tool_call_id = tool_call.get("id")     # ID định danh cho lần gọi này

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # Tìm hàm Python tương ứng và thực thi nó (Hành động thực tế)
        tool_to_use = tools_dict.get(tool_name)
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        # QUAN TRỌNG: Lưu lại lịch sử hội thoại để AI biết mình đã làm gì
        # 1. Lưu lại tin nhắn AI đã yêu cầu gọi Tool
        messages.append(ai_message)
        # 2. Lưu lại kết quả mà Tool đã trả về (Observation)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    # Nếu sau MAX_ITERATIONS mà vẫn chưa xong thì báo lỗi
    print("ERROR: Max iterations reached")
    return None

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    result = run_agent("What is the price of a laptop after applying a gold discount?")