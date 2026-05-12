import os
from typing import Any, Dict
from dotenv import load_dotenv

# --- IMPORT CHUẨN 2026 ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
# Sử dụng trực tiếp từ langgraph/prebuilt
from langgraph.prebuilt import create_react_agent

# 1. KHỞI TẠO CẤU HÌNH
load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
index_name = os.getenv("PINECONE_INDEX_NAME_OLLAMA")

vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# Khởi tạo LLM Gemini
llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite", temperature=0)

# 2. ĐỊNH NGHĨA CÔNG CỤ (TOOL)
@tool
def retrieve_context(query: str):
    """
    Tìm kiếm tài liệu liên quan từ Pinecone để trả lời câu hỏi.
    Dùng công cụ này khi cần kiến thức về LangChain hoặc tài liệu kỹ thuật.
    """
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

# 3. HÀM THỰC THI CHÍNH
def run_llm(query: str) -> Dict[str, Any]:
    tools = [retrieve_context]
    
    # Khởi tạo Agent
    agent_executor = create_react_agent(model=llm, tools=tools)

    input_messages = [
        ("system", "You are a helpful assistant. Use 'retrieve_context' to find info. Cite sources."),
        ("human", query)
    ]
    
    response = agent_executor.invoke({"messages": input_messages})
    
    # LẤY NỘI DUNG SẠCH TỪ TIN NHẮN CUỐI CÙNG
    last_message = response["messages"][-1]
    
    # Kiểm tra nếu content là list (dạng thô của Gemini) thì lấy phần text
    final_answer = ""
    if isinstance(last_message.content, list):
        for part in last_message.content:
            if isinstance(part, dict) and 'text' in part:
                final_answer += part['text']
            elif isinstance(part, str):
                final_answer += part
    else:
        final_answer = last_message.content

    # LẤY CONTEXT (Để hiển thị nguồn trong Sources)
    # Tìm trong lịch sử tin nhắn xem Tool nào đã được gọi
    context_docs = []
    for msg in response["messages"]:
        if hasattr(msg, "tool_names") or (hasattr(msg, "name") and msg.name == "retrieve_context"):
            # Ở đây chúng ta giả định context được lấy từ Tool message
            # Cách đơn giản nhất cho ông là query lại 1 lần nữa hoặc lấy từ state nếu dùng LangGraph phức tạp
            pass 

    return {
        "answer": final_answer,
        "context": [] # Tạm thời để trống nếu ông chưa lưu docs vào state
    }

# 4. CHẠY THỬ NGHIỆM
if __name__ == '__main__':
    print("🤖 Agent (LangGraph Stable) đang khởi động...")
    try:
        user_query = "What is a LangChain agent?"
        result = run_llm(query=user_query)
        
        print("\n" + "="*50)
        print(f"CÂU HỎI: {user_query}")
        print("-" * 20)
        print(f"TRẢ LỜI:\n{result['answer']}")
        print("="*50)
    except Exception as e:
        print(f"❌ Vẫn lỗi à? Thử gỡ rối: {e}")