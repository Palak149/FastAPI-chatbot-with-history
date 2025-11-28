# FastAPI Chatbot with Conversation History
 
Project Overview-
 
This project is a FastAPI-based chatbot that uses LangChain, Google Gemini, and a tool-router architecture to classify user messages and respond intelligently.
It includes:
 LLM (Gemini 2.5 Flash)
 Custom tools (positive, negative, marks, suicide-safe, general)
 Router-chain to classify intent
 Memory for chat history
 Frontend HTML + JavaScript chatbot UI
 Full API integration (CORS enabled)

1. High-Level System Architecture

The project is divided into two major layers:

1.1 Frontend (Client Layer)

Built using HTML, CSS, JavaScript

Displays chat interface

Sends user messages to backend via REST API

Receives chatbot responses

Handles UI updates, auto-scroll, message formatting

1.2 Backend (Server Layer)

Built using FastAPI

Provides REST endpoints: /chat and /history

Integrates with LangChain and Google Gemini 2.5 Flash

Maintains in-memory history

Implements:

Router (intent classifier)

Tools (logic modules)

Memory (chat history)

Delegation (branching logic)

Overall flow:
Frontend → Backend → LLM → Backend → Frontend

2. Backend Architecture
2.1 File Structure

FastAPI-chatbot-with-history/
│
├── backend/
│   ├── main.py              → FastAPI app + endpoints + CORS + history
│   ├── router_logic.py      → LLM, memory, tools, router, chat_agent()
│ 
├── frontend/
│   ├── index.html           → Chat UI + JavaScript API calls
│
├── .env
├── 3.0.0/                   → Python environment metadata (ignored by .gitignore)
│
├── requirements.txt         → All dependency packages
│
├── .gitignore               → Ignore venv, pycache, system files
│
└── README.md                → This documentation
├── runtime.txt

3. FastAPI Backend (main.py)
3.1 API Initialization

Creates FastAPI app instance

Configures CORS to allow frontend to communicate with backend

allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]

3.2 /chat Endpoint

Steps:

Receives JSON from frontend:

{ "message": "Hello" }


Validates message

Assigns a session_id if not provided

Calls:

chat_agent(user_input)


Saves details to HISTORY list:

{
  session_id,
  user_query,
  response,
  tool_used
}


Returns JSON response to frontend.

3.3 /history Endpoint

Returns complete conversation history:

{ "history": [ ... ] }

4. LLM Logic (router_logic.py)
4.1 LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


Used for:

Routing (classification)

Tool responses

General replies

4.2 Memory System

Uses LangChain ConversationBufferMemory

Stores:

User queries

Bot responses

Enables contextual conversations

Example stored history:

User: Hello
Bot: Hi there!
User: What are Alice’s marks?
Bot: Alice scored...

4.3 Tools (Logic Modules)

Tools are modular functions executed based on router decision.

positive_tool
Handles happy/positive messages.

negative_tool
Handles sad or negative messages.

marks_tool
Answers student mark queries using predefined JSON data.

suicide_tool_dynamic
Handles self-harm or critical messages in a safe and compliant way.

default_tool
Handles general conversation.

Each tool:

Uses chat history

Generates LLM prompt

Returns LLM response

4.4 Router (Intent Classifier)

Classifies user input into five categories:

positive

negative

marks

suicide

default

The router prompt instructs LLM to output only one category.

Example:

“I am very happy today.” → positive

“I am feeling sad.” → negative

“What are Bob’s marks?” → marks

“I want to harm myself.” → suicide

“Tell me a joke.” → default

4.5 Delegation (Routing Logic)

Uses LangChain RunnableBranch:

IF positive → positive_tool  
ELSE IF negative → negative_tool  
ELSE IF marks → marks_tool  
ELSE IF suicide → suicide_tool_dynamic  
ELSE → default_tool


This creates a clean, scalable routing system.

4.6 Main Orchestrator: chat_agent()

This function performs the entire AI processing pipeline.

Steps:

Loads chat history from memory

Router classifies the user input

Builds routing data

Executes selected tool

Saves conversation into memory

Returns structured result:

{
  "response": "...",
  "tool_used": "marks"
}

5. Frontend Architecture (index.html)
5.1 Responsibilities

Displays chat UI

Captures user input

Sends POST request to backend:

http://127.0.0.1:8000/chat


Displays bot response

Auto-scrolls chatbox

Shows which tool was used

Handles errors (backend down, network failure)

5.2 Fetch Request Structure
fetch("http://127.0.0.1:8000/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({ message: userMessage })
})

6. End-to-End Data Flow

Step-by-step sequence:

User types message in frontend

Frontend sends JSON to backend

Backend receives message

chat_agent() loads memory

LLM router classifies intent

Delegation selects appropriate tool

Tool generates LLM prompt and response

Memory stores the conversation

Backend stores entry in HISTORY

Backend returns JSON response

Frontend displays bot message

7. Key Advantages of This Architecture

Modular — Tools and router can be expanded easily

Scalable — New categories can be added without redesign

Safe — Suicide-tool ensures safety compliance

Contextual — Memory maintains conversation flow

Clean separation — Frontend and backend remain independent

Extensible — More tools or APIs can be plugged in as needed


 Folder Structure Explained
FastAPI-chatbot-with-history/
│
├── backend/

│   ├── venv/                → Virtual environment (ignored)
│   ├── main.py              → FastAPI app + endpoints + CORS + history
│   ├── router_logic.py      → LLM, memory, tools, router, chat_agent()
│ 
├── frontend/
│   ├── index.html           → Chat UI + JavaScript API calls
│
├── 3.0.0/                   → Python environment metadata (ignored by .gitignore)
│
├── requirements.txt         → All dependency packages
│
├── .gitignore               → Ignore venv, pycache, system files
│
└── README.md                → This documentation

 What Each File Contains
 backend/main.py

Starts FastAPI application

CORS setup (allows frontend to call backend)

/chat POST endpoint

Reads message from frontend

Calls chat_agent()

Stores conversation in memory

Returns LLM response

/history GET endpoint

Returns full conversation history

backend/router_logic.py

Contains the entire AI logic:

✓ LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

✓ Memory

Stores conversation:

ConversationBufferMemory()

✓ Tools

Custom functions used by the router:

positive_tool()

negative_tool()

marks_tool()

suicide_tool_dynamic()

default_tool()

✓ Router

LLM-based classifier that returns:

positive | negative | marks | suicide | default

✓ delegation_chain

Decides which tool will answer the user.

✓ chat_agent()

The core method:

Classifies input

Chooses proper tool

Saves memory

Returns response + tool name

Input → Output Processing Flow

 User types message in frontend →
JavaScript sends:

{
  "message": "How are you?"
}


 Backend receives message
Calls:

chat_agent(user_input)


 Router classifies it
Example:

positive


 Correct tool is executed
Example:

positive_tool()


 Result returned to frontend
Example response:

{
  "response": "I'm feeling great today!",
  "tool_used": "positive",
  "session_id": "abc123..."
}


 Frontend displays message in chatbox.

 How to Run Locally
1. Start backend
uvicorn backend.main:app --reload

2. Open frontend

Open frontend/index.html in any browser.

 Features Included

✔ LLM-based intent classification

✔ Multiple AI tools

✔ Conversation memory

✔ Suicide-safe responses

✔ Fully integrated frontend

✔ History endpoint

✔ Clean FastAPI architecture
