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

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,       # More memory for better answers
    n_threads=2,      # Safe for Render free tier
    n_batch=64,
    verbose=False
)

def generate_response(prompt: str):
    try:
        result = llm(
            prompt,
            max_tokens=256,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["User:", "Assistant:", "</s>"]
        )

        text = result["choices"][0]["text"]
        return text.strip()

    except Exception as e:
        return f"Model error: {str(e)}"




