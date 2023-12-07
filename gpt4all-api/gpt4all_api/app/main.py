import logging
import os
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from api_v1.api import router as v1_router
from api_v1.settings import settings
from dotenv import load_dotenv
from gpt4all import GPT4All
from fastapi.logger import logger as fastapi_logger

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title='GPT4All API', description='API for GPT4All')

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logger.info('Adding v1 endpoints...')
app.include_router(v1_router, prefix='/v1')

# Initialize model
@app.on_event("startup")
async def startup():
    global model
    inference_mode = os.getenv("INFERENCE_MODE", "cpu").lower()  # Default to CPU if not specified
    print(f"INFERENCE_MODE: {inference_mode}")
    model_path = os.path.join(settings.gpt4all_path, settings.model)

    if inference_mode not in ["gpu", "cpu"]:
        raise HTTPException(status_code=500, detail="Invalid INFERENCE_MODE specified")

    try:
        model = GPT4All(model_name=settings.model, model_path=settings.gpt4all_path, device=inference_mode)
        logger.info(f"Model initialized for {inference_mode.upper()}: {model_path}")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing model for {inference_mode.upper()}")

    logger.info("GPT4All API is ready for inference.")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down API")

# This is needed to get logs to show up in the app
if "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    gunicorn_logger = logging.getLogger("gunicorn")

    root_logger = logging.getLogger()
    fastapi_logger.setLevel(gunicorn_logger.level)
    fastapi_logger.handlers = gunicorn_error_logger.handlers
    root_logger.setLevel(gunicorn_logger.level)

    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.handlers = gunicorn_error_logger.handlers
else:
    # https://github.com/tiangolo/fastapi/issues/2019
    LOG_FORMAT2 = (
        "[%(asctime)s %(process)d:%(threadName)s] %(name)s - %(levelname)s - %(message)s | %(filename)s:%(lineno)d"
    )
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT2)
