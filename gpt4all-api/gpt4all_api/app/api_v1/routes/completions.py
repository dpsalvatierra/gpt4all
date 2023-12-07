import json
import logging
import time
from typing import List, Dict, Union, Optional, Iterable
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from gpt4all import GPT4All
from pydantic import BaseModel, Field
from api_v1.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Class definitions
class CompletionRequest(BaseModel):
    model: str = Field(settings.model, description='The model to generate a completion from.')
    prompt: Union[List[str], str] = Field(..., description='The prompt to begin completing from.')
    max_tokens: int = Field(settings.max_tokens, description='Max tokens to generate')
    temperature: float = Field(settings.temp, description='Model temperature')
    top_p: Optional[float] = Field(settings.top_p, description='top_p')
    top_k: Optional[int] = Field(settings.top_k, description='top_k')
    n: int = Field(1, description='How many completions to generate for each prompt')
    stream: bool = Field(False, description='Stream responses')
    repeat_penalty: float = Field(settings.repeat_penalty, description='Repeat penalty')

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: float
    finish_reason: str

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str
    object: str = 'text_completion'
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage

class CompletionStreamResponse(BaseModel):
    id: str
    object: str = 'text_completion'
    created: int
    model: str
    choices: List[CompletionChoice]

# Router and Endpoints
router = APIRouter(prefix="/completions", tags=["Completion Endpoints"])

def stream_completion(output: Iterable, base_response: CompletionStreamResponse):
    """
    Streams a GPT4All output to the client.
    """
    for token in output:
        chunk = base_response.copy()
        chunk.choices = [CompletionChoice(
            text=token,
            index=0,
            logprobs=-1,
            finish_reason=''
        )]
        yield f"data: {json.dumps(chunk.dict())}\n\n"

@router.post("/", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    '''
    Completes a GPT4All model response.
    '''
    if isinstance(request.prompt, list):
        if len(request.prompt) > 1:
            raise HTTPException(status_code=400, detail="Can only process one prompt per request.")
        else:
            request.prompt = request.prompt[0]

    # Adding inference_mode to the model
    model = GPT4All(model_name=request.model, model_path=settings.gpt4all_path, device=settings.inference_mode)

    output = model.generate(
        prompt=request.prompt or '',
        max_tokens=request.max_tokens or 500,
        temp=request.temperature or 0.7,
        top_k=request.top_k or 50,
        top_p=request.top_p or 0.9,
        streaming=request.stream
    )

    if request.stream:
        base_chunk = CompletionStreamResponse(
            id=str(uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[]
        )
        return StreamingResponse(stream_completion(output, base_chunk),
                                 media_type="text/event-stream")
    else:
        return CompletionResponse(
            id=str(uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[CompletionChoice(
                text=output,
                index=0,
                logprobs=-1,
                finish_reason='stop'
            )],
            usage=CompletionUsage(
                prompt_tokens=0,  # You may need to calculate these values
                completion_tokens=0,
                total_tokens=0
            )
        )

