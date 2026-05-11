# THAY ĐỔI 1: Thêm re (Regex) để trích xuất dữ liệu từ văn bản và inspect để đọc thông tin hàm Python.
import re
import inspect
from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# --- Tools ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Tra cứu giá của một sản phẩm trong danh mục."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Áp dụng mức giảm giá và trả về giá cuối cùng. Các mức: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price) # Đảm bảo giá là số thực để tính toán
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Lưu danh sách hàm vào dictionary để gọi bằng tên (string)
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# THAY ĐỔI 3: Xóa bỏ JSON Schema phức tạp. 
# Thay vào đó, ta dùng hàm này để tự động lấy tên hàm, tham số và mô tả (docstring) biến thành văn bản.
def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ giúp lấy hàm gốc nếu hàm đó đang bị bao bởi các decorator (như @traceable)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function) # Lấy tham số: (product: str)
        docstring = inspect.getdoc(tool_function) or ""  # Lấy phần mô tả hàm
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

# ĐÂY LÀ LINH HỒN CỦA AGENT: Prompt theo phong cách ReAct (Reason + Act)
# Nó ép AI phải suy nghĩ theo cấu trúc: Thought -> Action -> Action Input -> Observation
react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# THAY ĐỔI 4: Loại bỏ tham số 'tools=' trong ollama.chat. 
# Lúc này AI chỉ nghĩ nó đang viết văn bản bình thường, không biết mình là Agent.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

# --- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # Khởi tạo prompt gốc và scratchpad (nơi ghi lại lịch sử suy nghĩ/hành động)
    prompt = react_prompt.format(question=question)
    scratchpad = "" 

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        # Gộp prompt gốc với những gì AI đã làm ở các vòng trước
        full_prompt = prompt + scratchpad

        # 'stop': ["\nObservation"] cực kỳ quan trọng: Nó bắt AI dừng lại ngay khi viết xong Action Input.
        # Nếu không có nó, AI sẽ tự bịa ra kết quả (Observation) luôn.
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        # Bước 1: Dùng Regex tìm xem AI đã đưa ra câu trả lời cuối cùng chưa
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # THAY ĐỔI 6: Dùng Regex để bóc tách Action (Tên hàm) và Action Input (Tham số) từ văn bản thuần.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("  [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Xử lý chuỗi tham số (ví dụ: "laptop" hoặc "1299.99, gold") thành danh sách Python
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            # Thực thi hàm thực tế bằng cách giải nén danh sách tham số (*args)
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] {observation}")

        # THAY ĐỔI 7: Cập nhật scratchpad. Lịch sử lúc này là một chuỗi văn bản dài dần theo thời gian.
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")