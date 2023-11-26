import logging
import time
from typing import List, Optional
from uuid import uuid4
from fastapi import APIRouter
from pydantic import BaseModel, Field
from api_v1.settings import settings
from fastapi.responses import StreamingResponse

# Assuming gpt4all module is imported correctly
from gpt4all import GPT4All

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

### This should follow https://github.com/openai/openai-openapi/blob/master/openapi.yaml
class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(settings.model, description='The model to generate a completion from.')
    messages: List[ChatCompletionMessage] = Field(..., description='Messages for the chat completion.')
    max_tokens: int = Field(None, description='Max tokens to generate')
    temperature: float = Field(settings.temp, description='Model temperature')
    top_p: Optional[float] = Field(settings.top_p, description='top_p')
    top_k: Optional[int] = Field(settings.top_k, description='top_k')
    n: int = Field(1, description='How many completions to generate for each prompt')
    stream: bool = Field(False, description='Stream responses')
    repeat_penalty: float = Field(settings.repeat_penalty, description='Repeat penalty')

class ChatCompletionChoice(BaseModel):
    message: ChatCompletionMessage
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

router = APIRouter(prefix="/chat", tags=["Completions Endpoints"])

@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    '''
    Completes a GPT4All model response based on the chat messages.
    '''
    
    try:
        # Initialize GPT4All model
        model = GPT4All(model_name=request.model, model_path=settings.gpt4all_path)

        # Append messages to the chat session
        for message in request.messages:
            model.current_chat_session.append({
                "role": message.role,
                "content": message.content
            })

        if request.stream:
            # Streaming response
            output = model.generate(
                prompt="",
                max_tokens=200,  # Adjust as necessary
                temp=0.7,       # Adjust as necessary
                streaming=True,
                top_k=50,       # Adjust as necessary
                top_p=1.0,      # Adjust as necessary
                # Include other parameters as needed
            )
            base_response = ChatCompletionResponse(
                id=str(uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=[],
                usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # Placeholder values
            )
            return StreamingResponse(
                stream_completion(output, base_response),
                media_type="text/event-stream"
            )
        else:            
            response_content = model.generate(
                prompt="",
                max_tokens=200,  # Adjust as necessary
                temp=0.7,       # Adjust as necessary
                streaming=False, # Non-streaming response
                top_k=50,       # Adjust as necessary
                top_p=1.0,      # Adjust as necessary
                
                # Include other parameters as needed
            )
            response_message = ChatCompletionMessage(
                role="system",
                content=response_content if isinstance(response_content, str) else ""
            )
            response_choice = ChatCompletionChoice(
                message=response_message,
                index=0,
                logprobs=-1.0,  # Placeholder value
                finish_reason="length"  # Placeholder value
            )
            return ChatCompletionResponse(
                id=str(uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=[response_choice],
                usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # Placeholder values
            )
    except Exception as e:
        # Handle the error
        logger.error(f"An error occurred: {str(e)}")
        # Return an error response
        return {"error": "An error occurred"}
        
        
            
    else:
        # Non-streaming response
        response_content = model.generate(
            prompt="",
            max_tokens=200,  # Adjust as necessary
            temp=0.7,       # Adjust as necessary
            streaming=False, # Non-streaming response
            top_k=50,       # Adjust as necessary
            top_p=1.0,      # Adjust as necessary
            
            # Include other parameters as needed
        )
        response_message = ChatCompletionMessage(
            role="system",
            content=response_content if isinstance(response_content, str) else ""
        )
        response_choice = ChatCompletionChoice(
            message=response_message,
            index=0,
            logprobs=-1.0,  # Placeholder value
            finish_reason="length"  # Placeholder value
        )
        return ChatCompletionResponse(
            id=str(uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[response_choice],
            usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # Placeholder values
        )

def stream_completion(output):
    """
    Generator function to stream the output.
    """
    for token in output:
        yield f"data: {token}\n\n"