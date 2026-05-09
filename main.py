import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # Đảm bảo đã cài: uv pip install langchain-openai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv() 

def main():
    print("--- Đang xử lý dữ liệu ---")
    
    information = """
    Elon Musk sinh vào sáng lúc 7 giờ 30 phút, ngày 28 tháng 6 năm 1971 tại Pretoria...
    (Dữ liệu của bạn giữ nguyên)
    """

    summary_template = """
    Dựa vào thông tin dưới đây:
    {information}
    
    Hãy thực hiện:
    1. Một đoạn tóm tắt ngắn gọn về người đó.
    2. Liệt kê 2 điểm thú vị hoặc điểm mạnh của người đó.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )

    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.1-flash-lite", # Dùng đúng tên từ danh sách quét được
        temperature=0
    )
    chain = summary_prompt_template | llm

    # Invoke chain
    try:
        response = chain.invoke(input={"information": information})
        print("\nKẾT QUẢ:")
        print(response.content)
    except Exception as e:
        print(f"Lỗi rồi: {e}")

if __name__ == "__main__":
    main()
