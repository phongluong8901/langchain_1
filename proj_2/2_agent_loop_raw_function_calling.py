from dotenv import load_dotenv

# Nạp các biến môi trường từ file .env (thường chứa API Key cho LangSmith)
load_dotenv()

import ollama
from langsmith import traceable

# Cấu hình số lần lặp tối đa của Agent và tên Model AI sử dụng
MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---
# Các hàm Python thực thi các tác vụ cụ thể mà AI có thể gọi

@traceable(run_type="tool") # Gửi log của hàm này lên LangSmith để theo dõi
def get_product_price(product: str) -> float:
    """Tra cứu giá của một sản phẩm trong danh mục."""
    print(f"    >> Executing get_product_price(product='{product}')")
    # Giả lập database giá sản phẩm
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Áp dụng mức giảm giá vào giá gốc và trả về giá cuối cùng.
    Các mức giảm giá có sẵn: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    # Tính toán giá sau khi giảm và làm tròn 2 chữ số thập phân
    return round(price * (1 - discount / 100), 2)

# Khác biệt 2: Không có @tool, chúng ta phải định nghĩa JSON schema thủ công cho mỗi hàm.
# Đây chính xác là những gì decorator @tool của LangChain tự động tạo ra
# từ gợi ý kiểu dữ liệu (type hints) và docstring của hàm.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# LƯU Ý: Ollama cũng có thể tự tạo các schema này nếu bạn truyền hàm
# trực tiếp như một công cụ (tương tự như decorator @tool của LangChain):
#   tools_for_llm = [get_product_price, apply_discount]
# Tuy nhiên, điều này yêu cầu docstring của bạn phải tuân theo định dạng Google docstring
# để Ollama có thể phân tích mô tả tham số từ phần Args. Ví dụ:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# Chúng tôi giữ phiên bản JSON thủ công ở đây để bạn có thể thấy những gì @tool đang ẩn đi cho bạn.

# --- Helper: traced Ollama call ---
# Khác biệt 3: Không có LangChain, chúng ta phải thủ công theo dõi (trace) các cuộc gọi LLM cho LangSmith.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    # Gọi thư viện Ollama thuần để chat với đầy đủ tools và lịch sử tin nhắn
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# --- Agent Loop ---
# Vòng lặp điều khiển cách Agent suy nghĩ và thực hiện hành động


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # Bản đồ ánh xạ tên hàm từ AI sang hàm Python thực tế
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # Khởi tạo danh sách tin nhắn (bộ nhớ tạm thời của Agent)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Bắt đầu vòng lặp cho đến khi đạt mục tiêu hoặc quá số lần lặp
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Khác biệt 5: Gọi ollama.chat() trực tiếp thay vì llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        # Kiểm tra xem AI có yêu cầu gọi công cụ nào không
        tool_calls = ai_message.tool_calls

        # Nếu không có yêu cầu gọi công cụ, đây chính là câu trả lời cuối cùng
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Chỉ xử lý cuộc gọi công cụ ĐẦU TIÊN — ép buộc thực hiện một công cụ mỗi vòng lặp
        tool_call = tool_calls[0]
        # Khác biệt 6: Truy cập thuộc tính bằng dấu chấm (.function.name) thay vì truy cập kiểu dict (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # Tìm hàm tương ứng trong dictionary để chuẩn bị thực thi
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Khác biệt 7: Gọi trực tiếp hàm Python với các tham số từ AI (giải nén dictionary bằng **)
        observation = tool_to_use(**tool_args)


        print(f"  [Tool Result] {observation}")

        # Lưu lại tin nhắn của AI (yêu cầu gọi tool) vào lịch sử
        messages.append(ai_message)
        # Thêm kết quả thực thi công cụ (observation) vào lịch sử với vai trò là 'tool'
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    # Nếu thoát khỏi vòng lặp mà chưa có kết quả cuối cùng
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    # Kích hoạt Agent với câu hỏi cụ thể
    result = run_agent("What is the price of a laptop after applying a gold discount?")