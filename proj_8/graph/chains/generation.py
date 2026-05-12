# /graph/chains/generation.py

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Khởi tạo LLM (Sử dụng Qwen 3 bản 1.7b chạy Local qua Ollama)
llm = ChatOllama(
    model="qwen3:1.7b", 
    temperature=0 # Set bằng 0 để câu trả lời mang tính kỹ thuật, chính xác, ít "bay bổng"
)

# 2. Định nghĩa khuôn mẫu câu trả lời (Prompt Template)
# Đây là chỉ dẫn để AI biết nó phải làm gì với đống dữ liệu được cung cấp
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
Question: {question} 
Context: {context} 
Answer:"""

# Chuyển đổi chuỗi văn bản trên thành một đối tượng Prompt chuẩn của LangChain
prompt = ChatPromptTemplate.from_template(template)

# 3. Thiết lập Chuỗi sản xuất (Generation Chain)
# Luồng dữ liệu: Dữ liệu vào -> Điền vào Prompt -> Gửi cho LLM -> Chuyển kết quả thành chuỗi văn bản (String)
generation_chain = prompt | llm | StrOutputParser()