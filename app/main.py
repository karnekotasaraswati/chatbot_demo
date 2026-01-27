from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router

app = FastAPI()

# ✅ ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, OPTIONS, GET
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def home():
    return {"status": "AI Chatbot Running"}
