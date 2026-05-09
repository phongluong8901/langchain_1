import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from proj-1!")
    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("GOOGLE_API_KEY"))


if __name__ == "__main__":
    main()
