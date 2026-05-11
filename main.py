import os
from dotenv import load_dotenv

# Import các thành phần từ LangChain
from langchain_core.messages import HumanMessage # Đối tượng đại diện cho tin nhắn của người dùng
from langchain_core.prompts import ChatPromptTemplate # Công cụ tạo mẫu câu lệnh (prompt)
from langchain_pinecone import PineconeVectorStore # Kết nối với database Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # Dùng LLM và Embedding của Google

# Bước 1: Nạp các API Key từ file .env
load_dotenv()

print("Initializing components...")

# Bước 2: Khởi tạo mô hình Embedding của Google
# QUAN TRỌNG: Phải trùng model và số chiều (dimensions) với lúc bạn Ingest dữ liệu
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=512  # THÊM DÒNG NÀY: Ép về 512 để khớp với Index của bạn
)

# Bước 3: Khởi tạo mô hình ngôn ngữ (LLM) Gemini
# Dùng gemini-1.5-flash để tốc độ phản hồi nhanh và miễn phí
llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite", temperature=0)

# Bước 4: Kết nối với Index đã có trên Pinecone
vectorstore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"], 
    embedding=embeddings
)

# Bước 5: Tạo bộ truy xuất (Retriever)
# search_kwargs={"k": 3} nghĩa là lấy ra 3 đoạn văn bản liên quan nhất
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Bước 6: Định nghĩa mẫu câu lệnh cho RAG
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(docs):
    """Hàm phụ trợ để gộp các đoạn văn bản tìm được thành một chuỗi duy nhất."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Quy trình RAG thủ công (không dùng LCEL):
    1. Tìm kiếm -> 2. Ghép nội dung -> 3. Tạo câu hỏi -> 4. AI trả lời.
    """
    
    # Bước 7: Truy xuất các tài liệu liên quan từ Pinecone dựa trên câu hỏi
    docs = retriever.invoke(query)

    # Bước 8: Gộp nội dung các đoạn văn bản đó lại
    context = format_docs(docs)

    # Bước 9: Đưa ngữ cảnh và câu hỏi vào mẫu Prompt
    messages = prompt_template.format_messages(context=context, question=query)

    # Bước 10: Gửi toàn bộ dữ liệu cho Gemini để lấy câu trả lời cuối cùng
    response = llm.invoke(messages)

    return response.content

if __name__ == "__main__":
    print("Retrieving...")

    # Câu hỏi cần hỏi AI
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Cách 0: Hỏi thẳng AI (Không dùng dữ liệu trong Pinecone)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer (Kiến thức chung của AI):")
    print(result_raw.content)

    # ========================================================================
    # Cách 1: Sử dụng RAG (Tìm trong Pinecone trước rồi mới trả lời)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: With RAG (Using Pinecone Data)")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer (Dựa trên tài liệu blog của bạn):")
    print(result_without_lcel)