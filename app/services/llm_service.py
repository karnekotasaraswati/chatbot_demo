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


from llama_cpp import Llama

MODEL_PATH = "./models/model.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=1,
    n_batch=16,
    verbose=False
)

def generate_response(prompt: str):
    try:
        result = llm(
            prompt,
            max_tokens=80,
            temperature=0.6
        )

        return result["choices"][0]["text"]

    except Exception as e:
        return f"Model error: {str(e)}"
