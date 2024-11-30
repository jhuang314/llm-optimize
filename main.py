from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from functools import wraps
from http import HTTPStatus
import torch
import json
import os
import logging
from threading import Thread
from datetime import date
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Environment variables
ENABLE_AUTH = os.getenv('ENABLE_AUTH') in ('True', 'true')
API_TOKEN = os.getenv("LOCAL_API_TOKEN", "")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants and default configurations
_MODEL_DICT = {
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B-Instruct",
    "Llama-3.2-1B-4bit": "unsloth/Llama-3.2-1B-bnb-4bit",
    # "Llama-3.2-90B": "unsloth/Llama-3.2-90B-Vision-Instruct",
    "Gemma-2-27b-4bit": "unsloth/gemma-2-27b-bnb-4bit",
}
_MODELS = {
    "Llama-3.2-1B": None,
    "Llama-3.2-1B-4bit": None,
    # "": None,
    "Gemma-2-27b-4bit": None,
}

_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.7
_TOP_K = 1000
_TOP_P = 0.99


# Special Tokens
BEGIN_OF_TEXT = "<|begin_of_text|>"
END_OF_TURN = "<|eot_id|>"
START_HEADER_ID = "<|start_header_id|>"
END_HEADER_ID = "<|end_header_id|>"


logging.basicConfig(
    filename="llm_server.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger("llm server")

logger.info(f"environment variables. ENABLE_AUTH:{ENABLE_AUTH}, API_TOKEN:{API_TOKEN}")


def load_models():
    for model_id, model_name in _MODEL_DICT.items():
        logger.info(f"Loading model {model_id} from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        model = model.to(DEVICE)
        _MODELS[model_id] = (model, tokenizer)


def generate_stream(prompt, max_tokens, temperature, model_id):
    model, tokenizer = _MODELS.get(model_id, (None, None))
    if model is None:
        return "Invalid model parameter"

    input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Input Tokens: {input_ids['input_ids'].shape[-1]}")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_p=_TOP_P,
        top_k=_TOP_K,
        temperature=temperature,
        tokenizer=tokenizer,
        stop_strings=[END_OF_TURN, START_HEADER_ID, END_HEADER_ID]
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    tok_count = 0
    for token in streamer:
        tok_count += 1
        # Yield each token in OpenAI-compatible JSON format
        yield f"data: {json.dumps({'object': 'chat.completion.chunk','choices': [{'delta': {'content': token}, 'finish_reason': None} ]})}\n\n"

    logger.info(f"Output Tokens: {tok_count}")
    # Final message to signal end of streaming
    yield f"data: {json.dumps({'object': 'chat.completion.chunk','choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        logger.info(f"Request from: {request.remote_addr}")
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning("Missing or invalid token")
            return Response("Missing or invalid token", HTTPStatus.UNAUTHORIZED)

        token = auth_header.split(" ")[1]
        if ENABLE_AUTH and token != API_TOKEN:
            logger.warning("Unauthorized")
            return Response("Unauthorized", HTTPStatus.FORBIDDEN)

        logger.info("Authorized")
        return f(*args, **kwargs)
    return decorated

@app.route('/v1/completions/', methods=['POST'])
@requires_auth
def generate_completion():
    # Extract data from the request
    data = request.json
    prompt = BEGIN_OF_TEXT + data.get('prompt', '')
    logger.info(f"Prompt: {prompt}")
    max_tokens = min(data.get('max_tokens', 100), _MAX_TOKENS)
    logger.info(f"Request for {max_tokens} tokens")
    model_id = data.get('model',None)
    logger.info(f"Model: {model_id}")
    temperature = data.get('temperature', _DEFAULT_TEMPERATURE)
    if model_id is None:
        return Response("Missing model parameter", HTTPStatus.BAD_REQUEST)
    model, tokenizer = _MODELS.get(model_id, (None, None))
    if model is None:
        return Response("Invalid model parameter", HTTPStatus.BAD_REQUEST)

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Input Tokens: {inputs['input_ids'].shape[-1]}")
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
    logger.info(f"Output Tokens: {len(outputs[0])}")
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Format response similar to OpenAI's API
    response = {
        "choices": [
            {
                "text": completion,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ]
    }
    return jsonify(response)

@app.route('/v1/chat/completions', methods=['POST'])
@requires_auth
def generate_chat_completion():
    # Extract chat history and max_tokens from the request
    data = request.json
    messages = data.get('messages', [])
    max_tokens = min(data.get('max_tokens', 100), _MAX_TOKENS)
    model_id = data.get('model', None)
    stream = data.get('stream', False)
    temperature = data.get('temperature', _DEFAULT_TEMPERATURE)
    if model_id is None:
        return Response("Missing model parameter", HTTPStatus.BAD_REQUEST)
    model, tokenizer = _MODELS.get(model_id, (None, None))
    if model is None:
        return Response("Invalid model parameter", HTTPStatus.BAD_REQUEST)

    # Prepare prompt from messages (chat history)
    prompt = BEGIN_OF_TEXT
    for i, message in enumerate(messages):
        role = message.get('role')
        content = message.get('content')
        if i == 0 and role != "system":
            today = date.today().strftime("%B %d, %Y")
            prompt += f"{START_HEADER_ID}system{END_HEADER_ID}\n\nToday Date: {today}\nYou are a helpful assistant. You can answer questions about any topic.{END_OF_TURN}"
        if role and content:
            prompt += f"{START_HEADER_ID}{role}{END_HEADER_ID}\n\n {content}{END_OF_TURN}"
    prompt += f"{START_HEADER_ID}{END_HEADER_ID}\n\n"

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Request for {max_tokens} tokens")
    logger.info(f"Model: {model_id}")

    if stream:
        # Stream the completion in OpenAI-compatible format
        return Response(generate_stream(prompt, max_tokens, temperature ,model_id), content_type="text/event-stream")
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    logger.info(f"Input Tokens: {inputs['input_ids'].shape[-1]}")
    outputs = model.generate(inputs['input_ids'], max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
    logger.info(f"Output Tokens: {len(outputs[0])}")
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Format response similar to OpenAI's API
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": completion
                },
                "index": 0,
                "finish_reason": "length"
            }
        ]
    }
    return jsonify(response)

load_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
