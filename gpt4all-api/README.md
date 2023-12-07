# GPT4All REST API
This directory contains the source code to run and build docker images that run a FastAPI app
for serving inference from GPT4All models. The API matches the OpenAI API spec.

The GPT4ALL REST API runs on port 4891 and supports CPU and GPU inference. Just add the model and inference mode (CPU/GPU), the model will download automagically in `gpt4all-api/gpt4all_api/models` directory

## Tutorial

The following tutorial assumes that you have checked out this repo and cd'd into it.

### Starting the app

First change your working directory to `gpt4all/gpt4all-api`.

Now you can build the FastAPI docker image. You only have to do this on initial build or when you add new dependencies to the requirements.txt file:
```bash
DOCKER_BUILDKIT=1 docker build -t gpt4all_api --progress plain -f gpt4all_api/Dockerfile.buildkit .
```

Then, start the backend with:

```bash
docker compose up --build
```

This will run both the API and locally hosted GPU inference server. If you want to run the API without the GPU inference server, you can run:

```bash
docker compose up --build gpt4all_api
```

To run the API with the GPU inference server, you will need to include environment variables (like the `MODEL_ID`). Edit the `.env` file and run
```bash
docker compose --env-file .env up --build
```


#### Spinning up your app
Run `docker compose up` to spin up the backend. Monitor the logs for errors in-case you forgot to set an environment variable above.


#### Development
Run

```bash
docker compose up --build
```
and edit files in the `api` directory. The api will hot-reload on changes.

You can run the unit tests with

```bash
make test
```

#### Viewing API documentation

Once the FastAPI ap is started you can access its documentation and test the search endpoint by going to:
```
localhost:4891/docs
```

#### Running inference

##### Completion
```python
import openai
openai.api_base = "http://localhost:4891/v1"

openai.api_key = "not needed for a local LLM"


def test_completion():
    model = "llama-2-7b-chat.Q4_0"
    prompt = "Who is Michael Jordan?"
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=50,
        temperature=0.28,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )
    assert len(response['choices'][0]['text']) > len(prompt)
    print(response)
```
##### Chat Completion
```python
import openai

# Ask user if they want to stream the response or not, y is True, n is False, handle errors
streaming = input("Do you want to stream the response? (y/n): ")

if streaming == 'y':
    streaming = True
elif streaming == 'n':
    streaming = False
else:
    print("Invalid input, please try again")
    exit()



# Max output tokens, default is 200
max_output_tokens = 200

# Modify OpenAI's API key and API base to use LLM's API server.
openai.api_key = 'Not needed for this LLM' # Please keep this variable empty
openai.api_base = 'http://127.0.0.1:4891/v1'
completion = openai.ChatCompletion.create(
    model="llama-2-7b-chat.Q4_0",
    max_tokens= max_output_tokens,
    messages =  [{'role': 'system', 'content': f'You are a helpful assistant who needs to anser in less than {max_output_tokens} tokens'},
                 {'role': 'assistant', 'content': 'Michael Jordan was a basketball player for the Chicago Bulls'},
                 {'role': 'user', 'content': 'Who is Michael Jordan?'}],
    stream=streaming)


# Print the response
if streaming:
    for chunk in completion:
        if ('role' in chunk['choices'][0]['delta'] and chunk['choices'][0]['delta']['role'] != None):
            print(chunk['choices'][0]['delta']['role'] + ': ', end='')
        else:
            print(chunk['choices'][0]['delta']['content'], end='')
else:
    print(completion['choices'][0]['message']['role'] + ': ' + completion['choices'][0]['message']['content'])

```