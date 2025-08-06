import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    USE_REMOTE_EMBEDDINGS = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

settings = Settings()
