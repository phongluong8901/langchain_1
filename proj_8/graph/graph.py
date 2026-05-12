from dotenv import load_dotenv # Nạp biến môi trường từ file .env (chứa API Keys)

# Các thư viện cốt lõi của LangGraph để dựng sơ đồ tư duy cho AI
from langgraph.graph import END, StateGraph 

# Import các 'Chains' - Đây là các module logic xử lý bằng LLM (Bộ não con)
from graph.chains.answer_grader import answer_grader # QC: Kiểm tra câu trả lời có khớp câu hỏi không
from graph.chains.hallucination_grader import hallucination_grader # QC: Kiểm tra AI có nói sạo không
from graph.chains.router import question_router, RouteQuery # Bộ điều hướng: Chọn nguồn dữ liệu đầu vào
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH # Các hằng số định danh trạm (Node)
from graph.nodes import generate, grade_documents, retrieve, web_search # Các hàm thực thi tại mỗi trạm
from graph.state import GraphState # Cấu trúc 'giỏ hàng' dữ liệu đi xuyên suốt sơ đồ

load_dotenv() # Kích hoạt nạp API Keys

# --- PHẦN 1: CÁC HÀM ĐIỀU KIỆN (CONDITIONAL EDGES) ---
# Giống như các rơ-le trong mạch điện, quyết định dòng dữ liệu đi hướng nào

def decide_to_generate(state):
    """Quyết định: Sau khi lọc tài liệu xong thì đi viết bài hay đi search Google tiếp"""
    print("---ASSESS GRADED DOCUMENTS---")

    # Nếu trong State đánh dấu là cần search web (do tài liệu nội bộ không đủ)
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, INCLUDE WEB SEARCH---")
        return WEBSEARCH # Trả về tên trạm tiếp theo là WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE # Đủ dữ liệu rồi, cho đi viết câu trả lời luôn

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """Hàm QC cuối cùng: Kiểm tra chất lượng câu trả lời trước khi xuất xưởng"""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Bước QC 1: Gọi hallucination_grader để check xem AI có tự bịa thông tin ngoài tài liệu không
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score: # Nếu kết quả là 1 (Không ảo giác)
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        
        # Bước QC 2: Gọi answer_grader check xem câu trả lời có đúng trọng tâm người dùng hỏi không
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful" # Đạt chuẩn -> Kết thúc (END)
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful" # Lạc đề -> Bắt đi search Web tìm thêm
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED, RE-TRY---")
        return "not supported" # Có ảo giác -> Bắt quay lại trạm GENERATE để viết lại

def route_question(state: GraphState) -> str:
    """Bộ định tuyến tại cổng vào: Xem câu hỏi này nên tra cứu ở đâu"""
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # LLM sẽ phân tích câu hỏi và trả về 'websearch' hoặc 'vectorstore'
    source: RouteQuery = question_router.invoke({"question": question})
    
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

# --- PHẦN 2: XÂY DỰNG SƠ ĐỒ (WORKFLOW) ---

workflow = StateGraph(GraphState) # Khởi tạo Graph với trạng thái GraphState

# 1. Khai báo các trạm (Nodes) có mặt trong hệ thống
workflow.add_node(RETRIEVE, retrieve) # Trạm rút trích dữ liệu từ VectorDB
workflow.add_node(GRADE_DOCUMENTS, grade_documents) # Trạm thẩm định tài liệu lấy ra
workflow.add_node(GENERATE, generate) # Trạm dùng LLM để viết câu trả lời
workflow.add_node(WEBSEARCH, web_search) # Trạm tra cứu Google Search

# 2. Thiết lập điểm bắt đầu có điều kiện (Router)
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

# 3. Kết nối các trạm bằng các cạnh (Edges)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS) # Lấy data xong mặc định đi lọc

# Cạnh có điều kiện sau bước lọc tài liệu
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH, # Nếu thiếu data -> rẽ sang trạm Web Search
        GENERATE: GENERATE,   # Nếu đủ data -> rẽ sang trạm viết bài
    },
)

# Cạnh có điều kiện sau bước viết bài (Vòng lặp QC)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE, # Ảo giác -> Quay lại trạm Generate viết lại
        "useful": END,            # Hoàn hảo -> Ra cổng kết thúc
        "not useful": WEBSEARCH,   # Chưa đủ ý -> Đi search Web thêm
    },
)

# Các đường dây bổ trợ
workflow.add_edge(WEBSEARCH, GENERATE) # Search Web xong thì đưa data về trạm viết bài

# --- PHẦN 3: BIÊN DỊCH VÀ XUẤT FILE ---

app = workflow.compile() # Biên dịch toàn bộ logic thành một ứng dụng hoàn chỉnh

# Vẽ sơ đồ logic ra file ảnh để kiểm tra luồng đi có đúng thiết kế không
app.get_graph().draw_mermaid_png(output_file_path="graph.png")