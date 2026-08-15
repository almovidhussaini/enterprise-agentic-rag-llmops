import os

from dotenv import load_dotenv


load_dotenv()


class Settings:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    LANGSMITH_API_KEY = os.getenv(
        "LANGSMITH_API_KEY"
    )

    LANGCHAIN_PROJECT=os.getenv(
        "LANGCHAIN_PROJECT"
    )

    MODEL_NAME = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile"
    )


settings = Settings()