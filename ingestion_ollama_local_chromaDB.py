import asyncio
import os
import shutil
from typing import List
from dotenv import load_dotenv

# Thư viện mới
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings # Đổi ở đây
from langchain_tavily import TavilyCrawl
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# --- CẤU HÌNH ---
DB_DIR = "chroma_db_ollama"
# Ollama mặc định chạy ở http://localhost:11434
embeddings = OllamaEmbeddings(model="nomic-embed-text")

async def main():
    log_header("OLLAMA AGENTIC PIPELINE")

    # 1. DỌN DẸP
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            log_warning(f"♻️  Đã xóa database cũ tại {DB_DIR}")
        except Exception as e:
            log_error(f"❌ Lỗi Permission: {e}")
            return

    # 2. CRAWL DỮ LIỆU
    tavily = TavilyCrawl()
    log_info("🌐 Đang quét tài liệu LangChain từ internet...")
    try:
        res = tavily.invoke({
            "url": "https://python.langchain.com/",
            "max_depth": 1,
            "extract_depth": "advanced"
        })
    except Exception as e:
        log_error(f"❌ Crawl lỗi: {e}")
        return

    # 3. XỬ LÝ VĂN BẢN
    all_docs = []
    for item in res.get("results", []):
        if "${" in item["url"] or len(item.get("raw_content", "")) < 300:
            continue
        all_docs.append(Document(page_content=item["raw_content"], metadata={"source": item["url"]}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_docs = splitter.split_documents(all_docs)
    log_success(f"✂️  Đã chuẩn bị xong {len(final_docs)} mảnh dữ liệu.")

    # 4. NHÚNG VÀO DATABASE (OLLAMA)
    log_info(f"🚀 Bắt đầu nạp vào ChromaDB bằng Ollama...")
    try:
        # Với Ollama, ông KHÔNG CẦN batch size nhỏ hay chờ đợi lâu nữa
        # Vì nó chạy local, cứ đẩy vào là xong!
        vectorstore = Chroma.from_documents(
            documents=final_docs,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        log_success(f"🎉 THÀNH CÔNG! Đã nạp {len(final_docs)} mảnh vào {DB_DIR}")
    except Exception as e:
        log_error(f"❌ Lỗi khi nạp database: {e}")

    log_header("PIPELINE COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())