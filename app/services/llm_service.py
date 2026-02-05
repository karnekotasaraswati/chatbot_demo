# from llama_cpp import Llama

# MODEL_PATH = "./models/model.gguf"


# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=4096,
#     n_threads=8
# )

# def generate_response(prompt: str) -> str:
#     output = llm(
#         prompt,
#         max_tokens=512,
#         temperature=0.3,
#         stop=["</s>"]
#     )

#     return output["choices"][0]["text"].strip()


# from llama_cpp import Llama

# MODEL_PATH = "./models/model.gguf"

# llm = Llama(
#     model_path=MODEL_PATH,
#     n_ctx=256,
#     n_threads=1,
#     n_batch=8,
#     verbose=False
# )

# def generate_response(prompt: str):
#     try:
#         result = llm(
#             prompt,
#             max_tokens=40,
#             temperature=0.6
#         )

#         return result["choices"][0]["text"]

#     except Exception as e:
#         return f"Model error: {str(e)}"



from llama_cpp import Llama

MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

_llm = None  # global cache


def load_llm():
    global _llm

    if _llm is None:
        print("Loading LLM model (Low memory mode)...")

        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,     # reduced context size (low RAM)
            n_threads=6,    # single thread (Render CPU safe)
            n_batch=512,     # small batch to reduce memory
            verbose=False
        )

        print("LLM loaded successfully")

    return _llm


def generate_response(prompt: str):
    try:
        llm = load_llm()

        stream = llm(
            prompt,
            max_tokens=120,
            temperature=0.3,
            repeat_penalty=1.1,
            stop=["User:", "Assistant:", "User Question:", "Context:", "Instruct:", "\nUser", "\nContext"],
            stream=True
        )

        for output in stream:
            yield output["choices"][0]["text"]

    except Exception as e:
        yield f"Model error: {str(e)}"
