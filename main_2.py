import os
from operator import itemgetter
from dotenv import load_dotenv

# Import các thành phần core của LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Nạp API Key
load_dotenv()

# 2. Cấu hình Model và Vector Store
# Khởi tạo bộ não (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

# Khởi tạo bộ nhúng (Embedding) - PHẢI GIỐNG HỆT lúc Ingest
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=512
)

# Kết nối tới database Pinecone đã nạp 37 mảnh
vectorstore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"], 
    embedding=embeddings
)

# 3. Tạo Retriever (Bộ truy xuất)
# search_kwargs={"k": 5} nghĩa là lấy 5 mảnh văn bản giống câu hỏi nhất
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 4. Định nghĩa Prompt (Mẫu câu hỏi)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt_template = ChatPromptTemplate.from_template(template)

# Hàm phụ trợ để gộp các mảnh văn bản thành 1 chuỗi dài
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Xây dựng Chain bằng LCEL (Đây là phần bạn cần nhất)
def create_retrieval_chain_with_lcel():
    """
    Giải thích luồng dữ liệu (Data Flow):
    1. Nhận vào dict {"question": "..."}
    2. RunnablePassthrough.assign: Thêm một biến 'context' vào dict đó.
    3. 'context' được tạo ra bằng cách: Lấy question -> Truy xuất (Retriever) -> Định dạng (format_docs).
    4. Toàn bộ dict {"question": "...", "context": "..."} được đẩy vào Prompt.
    5. Prompt đẩy sang LLM (Gemini).
    6. Kết quả LLM được parser chuyển về dạng chuỗi (String).
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    query = "What is a vector database and why should I know it?"
    
    print(f"User Question: {query}")
    print("-" * 30)
    
    # Khởi tạo và chạy Chain
    chain = create_retrieval_chain_with_lcel()
    result = chain.invoke({"question": query})
    
    print("\nAI Answer (RAG):")
    print(result)