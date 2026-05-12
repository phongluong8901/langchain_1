# /graph/nodes/web_search.py

from typing import Any, Dict

from langchain_core.documents import Document # Chuyển dữ liệu web về định dạng Document chuẩn của LangChain
from langchain_tavily import TavilySearch # Công cụ tìm kiếm chuyên dụng cho AI (Tavily)

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv() # Nạp API Key (TAVILY_API_KEY)

# Khởi tạo công cụ tìm kiếm, giới hạn lấy 3 kết quả tốt nhất để tránh làm LLM bị "loãng" thông tin
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Nhiệm vụ: Khi dữ liệu nội bộ không đủ, trạm này sẽ "lên mạng" tìm kiếm câu trả lời,
    sau đó gói dữ liệu đó lại và bỏ vào giỏ hàng chung.
    """
    print("---WEB SEARCH---") # In ra để ông biết Agent đang dùng đến "quyền trợ giúp" từ internet
    
    # 1. Lấy dữ liệu hiện tại từ giỏ hàng
    question = state["question"]
    documents = state["documents"] # Danh sách tài liệu hiện có (có thể đang trống hoặc ít)

    # 2. Thực hiện tra cứu trên Internet qua Tavily
    # tavily_results trả về một list các dictionary chứa 'content', 'url',...
    tavily_results = web_search_tool.invoke({"query": question})['results']
    
    # 3. Gom tất cả nội dung tìm được thành một chuỗi văn bản duy nhất
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    
    # 4. Chuyển đổi chuỗi văn bản thô này thành một đối tượng Document chuẩn
    # Điều này giúp các trạm sau (như Generate) xử lý đồng nhất với dữ liệu từ DB
    web_results = Document(page_content=joined_tavily_result)
    
    # 5. Cập nhật vào danh sách tài liệu đang có
    if documents is not None:
        documents.append(web_results) # Nếu đã có tài liệu từ DB thì cộng thêm vào
    else:
        documents = [web_results] # Nếu chưa có gì thì tạo danh sách mới
        
    # 6. Trả về State mới đã được bổ sung kiến thức từ web
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    # Đoạn này để ông test nhanh file này độc lập
    web_search(state={"question": "agent memory", "documents": None})