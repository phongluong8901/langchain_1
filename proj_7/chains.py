import datetime  # Thư viện để lấy thời gian thực (giúp AI biết hôm nay là ngày nào)
from dotenv import load_dotenv

load_dotenv()  # Tải các API Key từ file .env

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,  # Trình phân tích kết quả trả về từ Tool dạng JSON
    PydanticToolsParser,    # Trình phân tích kết quả trả về dựa trên Class Pydantic (đảm bảo kiểu dữ liệu)
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# Import các Schema (khuôn mẫu dữ liệu) từ file schemas.py
from schemas import AnswerQuestion, ReviseAnswer 

# 1. Khởi tạo mô hình và bộ phân tích dữ liệu
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite", 
    temperature=0
)

parser = JsonOutputToolsParser(return_id=True)
# parser_pydantic: Đảm bảo dữ liệu trả về phải khớp hoàn toàn với class AnswerQuestion
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# 2. Xây dựng Prompt mẫu cho "Chuyên gia nghiên cứu" (Actor)
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
            # Prompt này yêu cầu AI: 1. Làm theo chỉ dẫn, 2. Tự chê bai bản thân thật gắt gao, 3. Đề xuất từ khóa tìm kiếm.
        ),
        MessagesPlaceholder(variable_name="messages"), # Lưu lịch sử hội thoại
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    # .partial giúp tự động điền thời gian hiện tại mỗi khi prompt được gọi
    time=lambda: datetime.datetime.now().isoformat(),
)

# 3. Tạo Responder (Người trả lời lần đầu)
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# bind_tools: Ép AI phải sử dụng "công cụ" AnswerQuestion để trả lời (thay vì trả lời văn bản tự do)
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# 4. Tạo Revisor (Người sửa đổi và cập nhật)
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations...
    - Add a "References" section...
    - Ensure it is not more than 250 words.
"""

# Revisor sẽ dùng schema ReviseAnswer để cập nhật thông tin dựa trên lời chê và dữ liệu mới
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# 5. Chạy chương trình
if __name__ == "__main__":
    # Câu hỏi về lĩnh vực AI-Powered SOC (Trung tâm điều hành an ninh mạng tự động)
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    
    # Thiết lập chuỗi thực thi (Chain) cho lần trả lời đầu tiên
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic # Chuyển kết quả từ AI thành đối tượng Python dễ xử lý
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res) # In ra kết quả đã được cấu trúc hóa (gồm câu trả lời, lời tự chê và từ khóa tìm kiếm)