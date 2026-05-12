import asyncio
import os
from typing import List
from dotenv import load_dotenv

# Thư viện Core
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl

# Thư viện Embedding & Vector Store
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# Logger của ông
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

async def run_cloud_pipeline():
    log_header("OLLAMA TO PINECONE CLOUD PIPELINE")

    # 1. KHỞI TẠO CẤU HÌNH
    # nomic-embed-text trả về vector 768 chiều
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    index_name = os.getenv("PINECONE_INDEX_NAME_OLLAMA")
    
    if not os.getenv("PINECONE_API_KEY") or not index_name:
        log_error("❌ Thiếu API Key hoặc Index Name trong file .env")
        return

    # 2. BƯỚC CRAWL (Lấy dữ liệu từ Internet)
    tavily = TavilyCrawl()
    log_info("🌐 Đang quét tài liệu LangChain...")
    try:
        # Lấy depth=1 cho nhanh và chính xác
        res = tavily.invoke({
            "url": "https://python.langchain.com/",
            "max_depth": 1,
            "extract_depth": "advanced"
        })
    except Exception as e:
        log_error(f"❌ Lỗi Crawl: {e}")
        return

    # 3. LÀM SẠCH VÀ CHIA NHỎ (Chunking)
    all_docs = []
    for item in res.get("results", []):
        content = item.get("raw_content", "")
        url = item.get("url", "")
        # Lọc bỏ link rác hoặc template lỗi
        if "${" in url or len(content) < 300:
            continue
        all_docs.append(Document(page_content=content, metadata={"source": url}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_docs = splitter.split_documents(all_docs)
    log_success(f"✂️  Đã chuẩn bị xong {len(final_docs)} mảnh dữ liệu.")

    # 4. ĐẨY LÊN PINECONE
    log_info(f"🚀 Bắt đầu nhúng và đẩy lên Cloud Pinecone (Index: {index_name})...")
    try:
        # PineconeVectorStore.from_documents sẽ tự động gọi Ollama để nhúng 
        # và đẩy kết quả lên server Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            documents=final_docs,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        log_success(f"🎉 THÀNH CÔNG! {len(final_docs)} mảnh đã nằm trên Pinecone.")
        log_info("Ông có thể lên trang app.pinecone.io để kiểm tra số lượng Vector.")

    except Exception as e:
        log_error(f"❌ Lỗi khi đẩy lên Cloud: {e}")
        log_warning("Gợi ý: Kiểm tra xem Index trên Pinecone đã tạo đúng Dimension (768) chưa?")

if __name__ == "__main__":
    asyncio.run(run_cloud_pipeline())