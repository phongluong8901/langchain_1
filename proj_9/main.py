import asyncio  # Thư viện để chạy các hàm bất đồng bộ (async/await)
import os       # Thư viện tương tác với hệ điều hành (đọc biến môi trường)

from dotenv import load_dotenv # Thư viện để đọc file .env (chứa API Key)
from langchain_core.messages import HumanMessage # Định dạng tin nhắn của người gửi là "Người"
from langchain_mcp_adapters.tools import load_mcp_tools # Bộ chuyển đổi quan trọng: Biến MCP Tools thành LangChain Tools
from langchain_google_genai import ChatGoogleGenerativeAI # LLM của Google (bạn có thể thay bằng ChatOpenAI)
from langgraph.prebuilt import create_react_agent # Hàm tạo nhanh một Agent có khả năng Suy nghĩ & Hành động (ReAct)
from mcp import ClientSession, StdioServerParameters # Các lớp định nghĩa kết nối MCP
from mcp.client.stdio import stdio_client # Client để nói chuyện với Server qua terminal (stdio)

# Nạp các biến môi trường từ file .env (ví dụ: GOOGLE_API_KEY)
load_dotenv()

# Khởi tạo mô hình ngôn ngữ lớn (mặc định là gpt-4o hoặc gpt-3.5)
llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite", temperature=0)

# Định nghĩa cách gọi MCP Server. 
# Ở đây là dùng lệnh "python" để chạy file "servers/math_server.py"
stdio_server_params = StdioServerParameters(
    command="python",
    args=["servers/math_server.py"],
)

async def main():
    # Khởi tạo kết nối vật lý (đường ống stdio) tới Math Server
    async with stdio_client(stdio_server_params) as (read, write):
        
        # Thiết lập phiên làm việc (Session) trên đường ống vừa mở
        async with ClientSession(read_stream=read, write_stream=write) as session:
            
            # Bắt tay (Handshake) với server để đảm bảo kết nối thông suốt
            await session.initialize()
            print("session initialized")
            
            # Dòng quan trọng: Tự động quét Server và lấy ra các hàm (add, multiply...) 
            # rồi chuyển chúng thành định dạng mà LangChain hiểu được.
            tools = await load_mcp_tools(session)

            # Tạo một Agent thông minh. 
            # Agent này biết dùng 'llm' để suy luận và dùng 'tools' để tính toán.
            agent = create_react_agent(llm, tools)

            # Gửi câu hỏi cho Agent dưới dạng một tin nhắn (HumanMessage)
            # 'ainvoke' là gọi agent theo kiểu bất đồng bộ (async)
            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 54 + 2 * 3?")]})
            
            # In ra nội dung tin nhắn cuối cùng trong danh sách (chính là câu trả lời của AI)
            print(result["messages"][-1].content)

# Điểm chạy chương trình
if __name__ == "__main__":
    # Kích hoạt vòng lặp sự kiện để chạy hàm main() bất đồng bộ
    asyncio.run(main())