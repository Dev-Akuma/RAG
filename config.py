import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    USE_REMOTE_EMBEDDINGS: bool = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
    HUGGINGFACE_API_KEY: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Match your env var name exactly

settings = Settings()
