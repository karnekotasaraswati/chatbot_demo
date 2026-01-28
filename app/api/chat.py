# from fastapi import APIRouter
# from pydantic import BaseModel
# from app.services.llm_service import generate_response
# # from app.services.retrieval_service import search_context

# router = APIRouter()

# class ChatRequest(BaseModel):
#     question: str

# @router.post("/chat")
# async def chat_bot(request: ChatRequest):


#     # context = search_context(request.question)
#     # print("Retrieved Context:", context)

#     prompt = f"""
# You are StarZopp AI Assistant.

# Answer ONLY using this context.
# If answer is not found say: I don't know.

# # Context:
# # {context}

# Question:
# {request.question}
# """

#     answer = generate_response(prompt)
#     print(answer)
#     return {"answer": answer}

from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_service import generate_response

router = APIRouter()


# Health check endpoint (for Render & monitoring)
@router.get("/health")
def health_check():
    return {"status": "StarZopp API running"}


# Request schema
class ChatRequest(BaseModel):
    question: str


# Chat endpoint
@router.post("/chat")
async def chat_bot(request: ChatRequest):

    prompt = f"""
You are StarZopp AI Assistant.

Question:
{request.question}
"""

    answer = generate_response(prompt)

    return {
        "question": request.question,
        "answer": answer
    }
