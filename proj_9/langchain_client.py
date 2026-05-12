# langchain_client.py
import asyncio
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Cài đặt API Key của bạn (hoặc set trong môi trường hệ thống)
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

async def main():
    # 1. Cấu hình để gọi Math Server
    server_params = StdioServerParameters(
        command="python",
        args=["math_server.py"], # Đảm bảo file này nằm cùng thư mục
    )

    # 2. Khởi tạo kết nối stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Lấy danh sách tools từ MCP Server
            mcp_tools = await session.list_tools()
            
            # 3. Chuyển đổi MCP Tools sang LangChain Tools + Cơ chế gọi hàm (Bridge)
            langchain_tools = []
            for t in mcp_tools.tools:
                # Tạo một hàm bao (wrapper) để Agent gọi được qua MCP session
                def make_tool_func(tool_name):
                    async def call_mcp_tool(tool_input):
                        # Nếu tool_input là dict thì truyền thẳng, nếu là string thì bọc lại
                        args = tool_input if isinstance(tool_input, dict) else {"a": tool_input}
                        result = await session.call_tool(tool_name, args)
                        return result.content
                    return call_mcp_tool

                langchain_tools.append(
                    Tool(
                        name=t.name,
                        func=make_tool_func(t.name), # Hàm bridge ở đây
                        description=t.description
                    )
                )

            # 4. Sử dụng Google Gemini Model
            llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite", temperature=0)

            # 5. Thiết lập Prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Bạn là một trợ lý toán học. Hãy sử dụng công cụ được cung cấp để tính toán."),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # 6. Khởi tạo Agent
            agent = create_tool_calling_agent(llm, langchain_tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=langchain_tools, verbose=True)

            # 7. Chạy thử nghiệm
            print("--- Đang hỏi Gemini ---")
            query = "Tính giúp tôi: 154 cộng với 232, sau đó nhân kết quả đó với 10."
            
            # Vì LangChain AgentExecutor chạy đồng bộ/bất đồng bộ tùy version, 
            # ta dùng invoke để chạy
            response = await agent_executor.ainvoke({"input": query})
            print("\nKết quả cuối cùng từ AI:", response["output"])

if __name__ == "__main__":
    # Chạy vòng lặp asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass