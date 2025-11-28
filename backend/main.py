# =========================================================
# FastAPI backend for Chatbot
# =========================================================
# Features:
# 1. Receives user messages from frontend via POST /chat
# 2. Sends messages to chat_agent (LLM + router)
# 3. Returns chatbot response along with the tool used and session ID
# 4. Stores conversation history in-memory (HISTORY)
# =========================================================

from fastapi import FastAPI, Request  # FastAPI framework and request object
from fastapi.middleware.cors import CORSMiddleware  # Middleware to allow cross-origin requests
from backend.router_logic import chat_agent  # Import the chat agent function (handles routing + LLM)
import uuid  # To generate unique session IDs for each conversation/session

# =========================================================
# Create FastAPI application instance
# =========================================================
app = FastAPI()

# =========================================================
# CORS configuration
# ---------------------------------------------------------
# Allows frontend (different domain/origin) to access this API
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  # List of allowed frontend URLs
        "https://fastapi-chatbot-with-history-1.onrender.com",
        "https://fastapi-chatbot-with-history.onrender.com",
        "*",  # Optional: allows all origins for testing (use cautiously in production)
    ],
    allow_credentials=True,  # Allows sending cookies or authentication headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# =========================================================
# In-memory conversation history
# ---------------------------------------------------------
# Stores all chat sessions during server runtime.
# Each entry contains:
#   - session_id: unique ID per conversation/session
#   - user_query: what user sent
#   - response: chatbot's LLM response
#   - tool_used: which internal tool handled the message
# ---------------------------------------------------------
HISTORY = []

# =========================================================
# Root endpoint: GET /
# ---------------------------------------------------------
# Health check endpoint to verify API is running
# Returns a simple JSON message
# =========================================================
@app.get("/")
def root():
    return {"message": "Chatbot API running"}

# =========================================================
# Chat endpoint: POST /chat
# ---------------------------------------------------------
# Receives user message and returns LLM response + tool used + session_id
# Steps:
#   1. Parse JSON body from request
#   2. Validate that user input exists
#   3. Generate a new session_id if not provided
#   4. Call chat_agent (routes message to proper tool and generates LLM response)
#   5. Store conversation in HISTORY
#   6. Return structured response
# =========================================================
@app.post("/chat")
async def chat(request: Request):
    # 1. Parse JSON body from frontend
    data = await request.json()
    user_input = data.get("message")  # Extract user message

    # 2. Validate user input
    if not user_input:
        return {"error": "Message is required"}

    # 3. Generate or reuse session_id
    # If frontend provides session_id, reuse it; else generate a new UUID
    session_id = data.get("session_id", str(uuid.uuid4()))

    # 4. Call chat_agent to get LLM response + tool_used
    # chat_agent handles routing (positive, negative, marks, default, suicide)
    result = chat_agent(user_input)

    # 5. Save conversation to in-memory HISTORY
    HISTORY.append({
        "session_id": session_id,
        "user_query": user_input,
        "response": result["response"],  # LLM-generated text
        "tool_used": result["tool_used"]  # Which tool handled this message
    })

    # 6. Return response to frontend
    # Includes session_id so frontend can maintain session continuity
    return {
        "response": result["response"],
        "tool_used": result["tool_used"],
        "session_id": session_id
    }

# =========================================================
# History endpoint: GET /history
# ---------------------------------------------------------
# Returns all stored conversation history (for debugging/testing)
# Each item contains:
#   - session_id
#   - user_query
#   - response
#   - tool_used
# =========================================================
@app.get("/history")
async def history():
    return {"history": HISTORY}
