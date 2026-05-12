from typing import Any, Dict, List

import streamlit as st # Thư viện để tạo giao diện Web nhanh cho Python

from backend.core import run_llm # Import hàm xử lý logic AI từ file backend của bạn


# 1. Hàm hỗ trợ định dạng nguồn tài liệu
def _format_sources(context_docs: List[Any]) -> List[str]:
    """Hàm này nhận vào danh sách các Document và trả về danh sách các đường link nguồn (URL)."""
    return [
        str((meta.get("source") or "Unknown")) # Lấy giá trị key 'source' trong metadata, nếu không có thì để 'Unknown'
        for doc in (context_docs or []) # Duyệt qua từng tài liệu trong context
        if (meta := (getattr(doc, "metadata", None) or {})) is not None # Trích xuất metadata một cách an toàn
    ]


# 2. Cấu hình trang Web
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered") # Đặt tiêu đề tab trình duyệt và căn giữa nội dung
st.title("LangChain Documentation Helper") # Hiển thị tiêu đề lớn trên trang

# 3. Thanh Sidebar (Thanh bên cạnh)
with st.sidebar:
    st.subheader("Session") # Tiêu đề phụ
    if st.button("Clear chat", use_container_width=True): # Tạo nút bấm xóa lịch sử chat
        st.session_state.pop("messages", None) # Xóa các tin nhắn trong bộ nhớ tạm
        st.rerun() # Tải lại trang để áp dụng thay đổi

# 4. Quản lý lịch sử trò chuyện (Chat History)
# Streamlit sẽ mất dữ liệu khi load lại trang, nên cần dùng st.session_state để lưu giữ tin nhắn
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", # Vai trò của người gửi (AI)
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [], # Danh sách nguồn ban đầu là rỗng
        }
    ]

# 5. Hiển thị lại các tin nhắn cũ mỗi khi trang được load lại
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): # Tạo khung chat (AI hoặc User)
        st.markdown(msg["content"]) # Hiển thị nội dung tin nhắn dạng Markdown
        if msg.get("sources"): # Nếu tin nhắn có chứa nguồn tài liệu
            with st.expander("Sources"): # Tạo một ô có thể đóng/mở để hiện nguồn
                for s in msg["sources"]:
                    st.markdown(f"- {s}") # Liệt kê từng nguồn

# 6. Xử lý khi người dùng nhập câu hỏi
prompt = st.chat_input("Ask a question about LangChain…") # Tạo ô nhập liệu ở dưới cùng
if prompt:
    # Lưu câu hỏi của người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"): # Hiển thị ngay câu hỏi của user lên màn hình
        st.markdown(prompt)

    # Xử lý phản hồi từ AI
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"): # Hiển thị icon xoay xoay đang xử lý
                # Gửi câu hỏi sang backend để AI xử lý (RAG: Retrieval Augmented Generation)
                result: Dict[str, Any] = run_llm(prompt) 
                
                # Trích xuất câu trả lời và nguồn từ kết quả trả về
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            # Hiển thị câu trả lời của AI
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            # Lưu câu trả lời của AI vào lịch sử để không bị mất khi rerun
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.") # Hiển thị thông báo lỗi nếu code backend gặp vấn đề
            st.exception(e) # Hiển thị chi tiết lỗi để debug