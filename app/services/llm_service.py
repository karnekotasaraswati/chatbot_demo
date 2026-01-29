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

MODEL_PATH = "./models/model.gguf"

_llm = None   # private global variable


def load_llm():
    global _llm

    if _llm is None:
        print("Loading LLM model...")

        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=2,
            verbose=False
        )

        print("LLM loaded successfully")

    return _llm


def generate_response(prompt: str):
    try:
        llm = load_llm()   # always guaranteed

        result = llm(
            prompt,
            max_tokens=120,
            temperature=0.2,
            repeat_penalty=1.2,
            stop=["Final Answer:", "User Question:"]
        )

        return result["choices"][0]["text"].strip()

    except Exception as e:
        return f"Model error: {str(e)}"
