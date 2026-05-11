import os
import uuid  # Thư viện này tạo ra chuỗi ký tự ngẫu nhiên (VD: a1-b2-c3...) để làm ID
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Nạp môi trường: Đọc các API Key từ file .env để code có quyền truy cập Google & Pinecone
load_dotenv()

if __name__ == "__main__":
    print("--- Bắt đầu Ingest theo kiểu Manual Force ---")
    
    # 2. Đọc file: Lấy toàn bộ nội dung từ file blog của bạn vào bộ nhớ máy tính
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    # 3. Chia nhỏ (Chunking): 
    # Vì AI không thể "nuốt" trọn 1 file dài, ta cắt nó thành các mẩu 600 ký tự.
    # RecursiveCharacterTextSplitter thông minh ở chỗ nó cố gắng cắt ở đoạn văn (\n\n) trước, 
    # nếu vẫn dài thì mới cắt ở dòng (\n), rồi mới đến dấu cách.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=50, # Giữ lại 50 ký tự cuối của đoạn trước cho đoạn sau để không mất ngữ cảnh
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(document)
    print(f"Đã chia thành {len(texts)} mảnh.")

    # 4. Nhúng (Embedding): 
    # Biến chữ thành "Vector" (dãy số). Dùng model Gemini của Google.
    # Output 512 phải khớp chính xác với Dimension bạn đã tạo trên Pinecone Index.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=512
    )

    index_name = os.environ["PINECONE_INDEX_NAME"]
    
    # 5. Kết nối Vector Store: Khai báo với LangChain rằng chúng ta sẽ làm việc với Index này.
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # 6. Dọn dẹp: Xóa sạch dữ liệu cũ để tránh việc tìm kiếm ra kết quả rác từ các lần chạy lỗi trước.
    print("Đang xóa sạch Index...")
    try:
        vector_store.delete(delete_all=True)
    except:
        # Nếu Index trống sẵn thì bỏ qua lỗi
        pass

    # 7. VÒNG LẶP ÉP BUỘC (Trái tim của bản sửa lỗi):
    # Thay vì nạp cả list (dễ bị lỗi gộp), ta nạp từng mảnh một.
    print("Bắt đầu nạp từng mảnh...")
    for i, chunk in enumerate(texts):
        # Tạo 1 cái ID hoàn toàn mới cho mảnh này. 
        # Đây là chìa khóa để Record count nhảy từ 1 lên 37.
        unique_id = str(uuid.uuid4())
        
        # Gọi lệnh add_documents cho DUY NHẤT 1 mảnh văn bản kèm 1 ID riêng.
        # Pinecone sẽ hiểu đây là 37 giao dịch riêng biệt và lưu vào 37 hàng khác nhau.
        vector_store.add_documents(documents=[chunk], ids=[unique_id])
        
        # In ra để bạn theo dõi tiến độ, tránh cảm giác máy bị treo.
        print(f"--> Đã nạp mảnh {i+1}/{len(texts)} - ID: {unique_id}")

    print("--- TẤT CẢ ĐÃ XONG! ---")