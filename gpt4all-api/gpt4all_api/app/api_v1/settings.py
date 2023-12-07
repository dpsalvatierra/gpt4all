from pydantic import BaseSettings
from dotenv import load_dotenv
import os 

load_dotenv()
class Settings(BaseSettings):
    app_environment = 'dev'
    model: str = os.getenv("MODEL_BIN")
    gpt4all_path: str = '/models'
    inference_mode: str = os.getenv("INFERENCE_MODE")
    sentry_dns: str = None
    max_tokens: int = 200
    temp: float = 0.18
    top_p: float = 1.0
    top_k: int = 50
    repeat_penalty: float = 1.18



settings = Settings()
