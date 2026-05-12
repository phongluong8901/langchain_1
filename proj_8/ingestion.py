import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
# Import thư viện Ollama mới
from langchain_ollama import OllamaEmbeddings

load_dotenv()
os.environ["USER_AGENT"] = "MyAgenticApp/1.0"

# 1. Danh sách nguồn dữ liệu
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 2. Cào dữ liệu
print("--- Đang cào dữ liệu từ web ---")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 3. Chia nhỏ văn bản
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 4. Khởi tạo Ollama Embedding (Chạy local trên máy ông)
# Model nomic-embed-text mặc định có 768 dimensions
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 5. Lưu trữ vào Vector Database (ChromaDB)
# XÓA THƯ MỤC .chroma CŨ TRƯỚC KHI CHẠY
print("--- Đang tạo Vector Database (Ollama Local) ---")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=local_embeddings,
    persist_directory="./.chroma",
)

# 6. Khởi tạo Retriever
print("--- Đang khởi tạo Retriever ---")
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=local_embeddings,
).as_retriever()

print("--- HOÀN THÀNH! Dữ liệu local đã sẵn sàng ---")