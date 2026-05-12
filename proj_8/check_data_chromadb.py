from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Khởi tạo lại Embedding (Phải giống hệt lúc nạp dữ liệu)
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Kết nối tới database đã lưu
vectorstore = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=local_embeddings
)

# 3. Kiểm tra tổng số lượng dữ liệu
try:
    # Lấy toàn bộ dữ liệu từ collection
    results = vectorstore.get()
    
    total_docs = len(results['ids'])
    print(f"--- THÔNG TIN DATABASE ---")
    print(f"Tổng số đoạn (chunks) hiện có: {total_docs}")
    
    if total_docs > 0:
        # In thử 2 đoạn đầu tiên để check nội dung
        print(f"\n--- XEM THỬ 2 ĐOẠN ĐẦU TIÊN ---")
        for i in range(min(2, total_docs)):
            print(f"\n[ID]: {results['ids'][i]}")
            print(f"[Nguồn]: {results['metadatas'][i].get('source', 'Unknown')}")
            print(f"[Nội dung]: {results['documents'][i][:200]}...")
            print("-" * 30)
            
        # 4. Chạy thử một câu truy vấn tìm kiếm (Similarity Search)
        print(f"\n--- TEST SEARCH THỬ ---")
        query = "What is task decomposition?"
        search_results = vectorstore.similarity_search(query, k=1)
        if search_results:
            print(f"Câu hỏi test: {query}")
            print(f"Kết quả tìm thấy nhất: {search_results[0].page_content[:200]}...")
    else:
        print("Database đang trống rỗng. Hãy chạy lại file ingestion.py nhé!")

except Exception as e:
    print(f"Lỗi khi truy cập Database: {e}")