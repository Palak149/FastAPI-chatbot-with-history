
# This file sets up a FastAPI backend for a chatbot.
# It handles:
#   1. Receiving chat messages from the frontend
#   2. Sending them to the chat agent (LLM + routing)
#   3. Returning responses along with tool used and session ID
#   4. Storing conversation history in-memory
# ------------------------------

from fastapi import FastAPI, Request  # FastAPI framework and HTTP request object
from fastapi.middleware.cors import CORSMiddleware  # Middleware to allow frontend requests from other origins
from backend.router_logic import chat_agent  # Import chat agent function (async)
import uuid  # To generate unique session IDs for tracking conversations

# ------------------------------
# FastAPI instance
# ------------------------------
app = FastAPI()  # Create FastAPI application instance

# ------------------------------
# CORS configuration
# ------------------------------
# This allows the frontend app to access this API from a different domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  # List of allowed frontend URLs
        "https://fastapi-chatbot-with-history-1.onrender.com",
        "https://fastapi-chatbot-with-history.onrender.com",
        "*",  # Optional: allows all origins for testing
    ],
    allow_credentials=True,  # Allows sending cookies or authentication headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ------------------------------
# In-memory conversation history
# ------------------------------
# Stores past conversations in a list.
# Each item will contain:
#   - session_id: unique conversation/session identifier
#   - user_query: what the user sent
#   - response: chatbot's response
#   - tool_used: which internal tool handled the query
HISTORY = []

# ------------------------------
# Root endpoint
# ------------------------------
@app.get("/")
def root():
    """
    GET /
    Simple health check endpoint to verify that the API is running.
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
    Receives user messages from frontend and returns chatbot response.
    Steps:
      1. Parse JSON body from request
      2. Validate user input
      3. Generate or use existing session ID
      4. Send message to async chat_agent
      5. Save conversation to HISTORY
      6. Return structured response
    """
    # Parse JSON body
    data = await request.json()
    user_input = data.get("message")  # Extract user message

    # Validate user input
    if not user_input:
        return {"error": "Message is required"}

    # Use existing session_id if provided, else generate a new unique one
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Call async chat_agent to get response
    # chat_agent handles routing to the correct tool and remembers conversation
    result = await chat_agent(user_input)

    # Save conversation to in-memory HISTORY
    HISTORY.append({
        "session_id": session_id,
        "user_query": user_input,
        "response": result["response"],
        "tool_used": result["tool_used"]
    })

    # Return chatbot response along with tool used and session ID
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
    Returns all conversations stored in the server memory.
    Each conversation contains:
      - session_id
      - user_query
      - chatbot response
      - tool_used
    Useful for testing or reviewing past chats.
    """
    return {"history": HISTORY}
