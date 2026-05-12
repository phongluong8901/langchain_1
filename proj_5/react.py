from dotenv import load_dotenv
from langchain_core.tools import tool
# Thay đổi thư viện từ OpenAI sang Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

# Chuyển sang dùng model Gemini (ví dụ gemini-1.5-flash hoặc gemini-1.5-pro)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite", 
    temperature=0
).bind_tools(tools)