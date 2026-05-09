from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# 1. Định nghĩa Schema đầu ra
class Source(BaseModel):
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used")

# 2. Khởi tạo LLM và ép định dạng đầu ra (Structured Output)
llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.1-flash-lite", # Dùng đúng tên từ danh sách quét được
        temperature=0
    )
# Ép Gemini phải trả về đúng định dạng AgentResponse
structured_llm = llm.with_structured_output(AgentResponse)

# 3. Định nghĩa Công cụ
tools = [TavilySearchResults(max_results=3)]

# 4. Khởi tạo Agent (Sử dụng LangGraph để tránh lỗi ImportError cũ)
agent = create_react_agent(llm, tools)

def main():
    print("--- Hello from langchain-course! ---")
    
    query = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
    
    # Chạy agent để lấy dữ liệu từ web
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    # Sau khi agent tìm xong, lấy nội dung cuối cùng đưa vào structured_llm để format
    final_content = result["messages"][-1].content
    structured_response = structured_llm.invoke(final_content)
    
    print("\n--- KẾT QUẢ ĐÃ ĐƯỢC FORMAT ---")
    print(f"Câu trả lời: {structured_response.answer}")
    print(f"Nguồn tham khảo: {structured_response.sources}")

if __name__ == "__main__":
    main()