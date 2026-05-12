import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from logger import log_header, log_info, log_success

# 1. Khởi tạo lại cấu hình giống hệt lúc nạp
DB_DIR = "chroma_db_ollama"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def check_my_data():
    log_header("KIỂM TRA DỮ LIỆU CHROMADB")

    if not os.path.exists(DB_DIR):
        print(f"❌ Không tìm thấy folder {DB_DIR}")
        return

    # 2. Kết nối tới database đã có
    vectorstore = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings
    )

    # 3. Đếm số lượng record
    # Lưu ý: Chroma bản mới dùng phương thức này để lấy collection
    collection_count = vectorstore._collection.count()
    log_success(f"📊 Tổng số mảnh (chunks) đang có: {collection_count}")

    if collection_count > 0:
        # 4. Lấy thử 1 mảnh xem nội dung là gì
        # get() sẽ trả về danh sách IDs, metadatas, và documents
        sample = vectorstore._collection.get(limit=1)
        
        log_info("📄 Nội dung thử nghiệm của 1 mảnh:")
        print("-" * 50)
        print(f"ID: {sample['ids'][0]}")
        print(f"Metadata: {sample['metadatas'][0]}")
        print(f"Content: {sample['documents'][0][:200]}...") # In 200 ký tự đầu
        print("-" * 50)
    else:
        log_info("⚠️ Database trống rỗng!")

if __name__ == "__main__":
    check_my_data()