import logging
import time
from typing import List, Optional
from uuid import uuid4
from fastapi import APIRouter
import asyncio
from pydantic import BaseModel, Field
from api_v1.settings import settings
from fastapi.responses import StreamingResponse
import json
# Assuming gpt4all module is imported correctly
from gpt4all import GPT4All

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Class Definitions
class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(settings.model, description='The model to generate a completion from.')
    messages: List[ChatCompletionMessage] = Field(..., description='Messages for the chat completion.')
    max_tokens: int = Field(settings.max_tokens, description='Max tokens to generate')
    temperature: float = Field(settings.temp, description='Model temperature')
    top_p: Optional[float] = Field(settings.top_p, description='top_p')
    top_k: Optional[int] = Field(settings.top_k, description='top_k')
    n: int = Field(1, description='How many completions to generate for each prompt')
    stream: bool = Field(False, description='Stream responses')
    repeat_penalty: float = Field(settings.repeat_penalty, description='Repeat penalty')

class ChatCompletionChoice(BaseModel):
    message: Optional[ChatCompletionMessage] = None
    delta: Optional[ChatCompletionDelta] = None
    index: int
    logprobs: float
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'text_completion'
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# Router and Endpoints
router = APIRouter(prefix="/chat", tags=["Completions Endpoints"])

@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    try:
        model = GPT4All(model_name=request.model, model_path=settings.gpt4all_path)
        for message in request.messages:
            model.current_chat_session.append({
                "role": message.role,
                "content": message.content
            })

        if request.stream:
            return StreamingResponse(event_generator(model, request), headers=stream_headers())
        else:
            response_content = model.generate(
                prompt=request.messages[-1].content,
                max_tokens=request.max_tokens or 500,
                temp=request.temperature or 0.7,
                top_k=request.top_k or 50,
                top_p=request.top_p or 1.0,
                repeat_penalty=request.repeat_penalty,
                streaming=False
            )
            response_message = ChatCompletionMessage(role="system", content=response_content)
            response_choice = ChatCompletionChoice(message=response_message, index=0, logprobs=-1.0, finish_reason="length")
            return ChatCompletionResponse(id=str(uuid4()), created=int(time.time()), model=request.model, choices=[response_choice], usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0))
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def stream_headers():
    return {"Cache-Control": "no-cache", "Content-Type": "text/event-stream", "Connection": "keep-alive"}

async def event_generator(model, request):
    try:
        accumulated_response = ""
        chunk_size = 60  # Define a suitable chunk size
        delay = 0.5  # Delay in seconds

        for token in model.generate(
            prompt=request.messages[-1].content,
            max_tokens=request.max_tokens,
            temp=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repeat_penalty=request.repeat_penalty,
            streaming=True
        ):
            accumulated_response += token
            if len(accumulated_response) >= chunk_size:
                response_format = {
                    "choices": [{
                        "delta": {
                            "content": accumulated_response
                        }
                    }]
                }
                yield f"data: {json.dumps(response_format)}\n\n"
                accumulated_response = ""  # Reset the accumulator after yielding
                await asyncio.sleep(delay)  # Introduce a delay

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: Error: {e}\n\n"