# 1. Import các thành phần cần thiết
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Để tạo mẫu câu lệnh (Prompt)
from langchain_google_genai import ChatGoogleGenerativeAI

# 2. Định nghĩa Prompt cho vai trò "Phê bình" (Reflection)
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", # Thiết lập đóng vai hệ thống
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
            # Câu lệnh trên bảo AI: "Bạn là một KOL nổi tiếng, hãy chấm điểm và đưa ra lời khuyên chi tiết để tweet dễ lên xu hướng."
        ),
        # MessagesPlaceholder: Nơi lưu trữ toàn bộ lịch sử hội thoại (để AI biết bài viết trước đó là gì mà chê)
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 3. Định nghĩa Prompt cho vai trò "Người viết" (Generation)
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", # Thiết lập đóng vai hệ thống
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
            # Câu lệnh trên bảo AI: "Bạn là trợ lý viết lách chuyên về công nghệ. Nếu nhận được lời chê, hãy viết lại bản mới tốt hơn."
        ),
        # MessagesPlaceholder: Giúp AI nhớ được các bản nháp cũ và lời chê để sửa đổi
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 4. Khởi tạo mô hình ngôn ngữ (LLM)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite", 
    temperature=0
)

# 5. Tạo các "Chuỗi thực thi" (Chains) bằng toán tử Pipe (|)
# generate_chain: Kết hợp mẫu câu lệnh Viết với AI
generate_chain = generation_prompt | llm

# reflect_chain: Kết hợp mẫu câu lệnh Chê với AI
reflect_chain = reflection_prompt | llm