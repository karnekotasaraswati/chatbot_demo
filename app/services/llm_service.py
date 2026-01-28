from llama_cpp import Llama

MODEL_PATH = "./models/model.gguf"


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8
)

def generate_response(prompt: str) -> str:
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.3,
        stop=["</s>"]
    )

    return output["choices"][0]["text"].strip()
