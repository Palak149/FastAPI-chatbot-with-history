from fastapi import FastAPI, Request  # Import FastAPI framework and Request object
from fastapi.middleware.cors import CORSMiddleware  # Middleware to handle cross-origin requests
from backend.router_logic import chat_agent  # Import our chatbot logic from router_logic.py
import uuid  # To generate unique session IDs for users

# ------------------------------
# Create FastAPI instance
# ------------------------------
app = FastAPI()

# ------------------------------
# Configure CORS (Cross-Origin Resource Sharing)
# ------------------------------
# CORS allows your frontend (running on a different domain/port) to call this API.
# In production, replace "*" with the actual URL of your deployed frontend for security.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fastapi-chatbot-with-history-1.onrender.com",
        "https://fastapi-chatbot-with-history.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# In-memory store for conversation history
# ------------------------------
# This keeps all past messages in this server session.
# Each item will contain: session_id, user query, chatbot response, and tool used.
HISTORY = []

# ------------------------------
# Root endpoint
# ------------------------------
@app.get("/")
def root():
    """
    GET /
    Simple endpoint to verify that API is running.
    Returns a JSON message.
    """
    return {"message": "Chatbot API running"}

# ------------------------------
# Chat endpoint
# ------------------------------
@app.post("/chat")
async def chat(request: Request):
    """
    POST /chat
    Handles user messages sent from frontend.
    Returns chatbot response, tool used, and a session ID.
    """
    # Parse JSON body of the request
    data = await request.json()
    user_input = data.get("message")  # Extract user message

    # Validate input
    if not user_input:
        return {"error": "Message is required"}

    # Use existing session_id if sent by frontend, otherwise generate a new unique one
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Call our chat agent to get response
    result = chat_agent(user_input)

    # Save the conversation in the HISTORY list
    # This allows us to track past queries and responses in-memory
    HISTORY.append({
        "session_id": session_id,
        "user_query": user_input,
        "response": result["response"],
        "tool_used": result["tool_used"]
    })

    # Return the chatbot's response along with the tool used and session_id
    # Frontend can reuse session_id for continuing the same conversation
    return {
        "response": result["response"],
        "tool_used": result["tool_used"],
        "session_id": session_id
    }

# ------------------------------
# History endpoint
# ------------------------------
@app.get("/history")
async def history():
    """
    GET /history
    Returns all past conversations stored in the server's memory.
    Each record contains:
        - session_id
        - user_query
        - chatbot response
        - tool_used
    Useful for testing or reviewing conversation history.
    """
    return {"history": HISTORY}
