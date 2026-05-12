import asyncio
import os
import ssl
import time
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilyCrawl

# Import các hàm log từ logger.py
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

# 1. Khởi tạo môi trường
load_dotenv() 

# Cấu hình SSL tránh lỗi certificate
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# 2. Cấu hình Embeddings & Vector Store
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=512
)

# Khởi tạo database (Chroma)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Công cụ crawl
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 10):
    """Xử lý nạp dữ liệu với cơ chế chống lỗi phản hồi rỗng."""
    log_header("VECTOR STORAGE PHASE")
    log_info(f"📚 Tổng số sau khi lọc: {len(documents)} mảnh | Batch size: {batch_size}", Colors.DARKCYAN)

    if not documents:
        log_error("Không có mảnh nào hợp lệ để lưu!")
        return

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    successful = 0
    for i, batch in enumerate(batches):
        batch_num = i + 1
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                log_info(f"🚀 Đang nạp đợt {batch_num}/{len(batches)} (Lần thử {attempt + 1})...")
                
                # Thực hiện nạp
                await vectorstore.aadd_documents(batch)
                
                log_success(f"✅ Đợt {batch_num} thành công!")
                successful += 1
                
                # Nghỉ 10s để duy trì Quota an toàn cho gói Free
                if batch_num < len(batches):
                    await asyncio.sleep(10) 
                break 

            except Exception as e:
                err_msg = str(e)
                log_error(f"❌ Lỗi tại đợt {batch_num}: {err_msg}")
                
                # Nếu lỗi do Rate Limit (429) hoặc lỗi Index (do API trả về rỗng)
                if "429" in err_msg or "index" in err_msg.lower():
                    wait_time = 30 * (attempt + 1)
                    log_warning(f"⚠️ Đang tạm nghỉ {wait_time}s để hồi phục API...")
                    await asyncio.sleep(wait_time)
                else:
                    break

    log_info(f"📊 Hoàn thành: {successful}/{len(batches)} đợt.")

async def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")

    # BƯỚC 1: CRAWL DỮ LIỆU
    log_info("🗺️ TavilyCrawl: Bắt đầu quét trang tài liệu...", Colors.PURPLE)
    try:
        res = tavily_crawl.invoke(
            {
                "url": "https://python.langchain.com/",
                "max_depth": 2,
                "extract_depth": "advanced",
            }
        )
    except Exception as e:
        log_error(f"Lỗi khi crawl: {e}")
        return

    all_docs = []
    for item in res.get("results", []):
        content = item.get("raw_content", "")
        # Lọc ngay từ bước crawl: bỏ qua các trang rỗng hoặc link rác
        if content and len(content.strip()) > 100:
            log_info(f"🌐 Lấy dữ liệu: {item['url']}")
            all_docs.append(
                Document(
                    page_content=content,
                    metadata={"source": item["url"]},
                )
            )

    if not all_docs:
        log_error("Không tìm thấy nội dung hợp lệ từ Tavily!")
        return

    # BƯỚC 2: CHUNKING & CLEANING (CỰC KỲ QUAN TRỌNG)
    log_header("DOCUMENT CHUNKING & CLEANING")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    raw_splitted_docs = text_splitter.split_documents(all_docs)
    
    # Lọc sạch các mảnh rác trước khi đưa vào Embedding
    splitted_docs = []
    for doc in raw_splitted_docs:
        clean_content = doc.page_content.strip()
        # Loại bỏ các mảnh quá ngắn hoặc chỉ chứa ký tự đặc biệt/link rác
        if len(clean_content) > 20 and not clean_content.startswith("${"):
            splitted_docs.append(doc)
            
    log_success(f"✂️ Đã chia nhỏ và loại bỏ mảnh rác. Còn lại: {len(splitted_docs)} mảnh.")

    # BƯỚC 3: LƯU VÀO DATABASE
    await index_documents_async(splitted_docs, batch_size=10)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Hệ thống đã sẵn sàng!")

if __name__ == "__main__":
    # Lưu ý: Xóa folder chroma_db trước khi chạy lại nếu muốn sạch index
    asyncio.run(main())