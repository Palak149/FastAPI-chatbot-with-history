from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.router_logic import chat_agent
import uuid

app = FastAPI()

# CORS so frontend can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory history store
HISTORY = []

# Root
@app.get("/")
def root():
    return {"message": "Chatbot API running"}

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message")

    if not user_input:
        return {"error": "Message is required"}

    # Assign a session_id if not provided
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Get chatbot response
    result = chat_agent(user_input)

    # Save in history
    HISTORY.append({
        "session_id": session_id,
        "user_query": user_input,
        "response": result["response"],
        "tool_used": result["tool_used"]
    })

    # Return response + session_id so frontend can reuse
    return {
        "response": result["response"],
        "tool_used": result["tool_used"],
        "session_id": session_id
    }

# History endpoint
@app.get("/history")
async def history():
    return {"history": HISTORY}
