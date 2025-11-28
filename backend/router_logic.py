# This file handles the main chatbot logic:
#   1. LLM setup using Google Gemini (via langchain_google_genai)
#   2. Conversation memory
#   3. Tools for different intents (happy, sad, marks, suicide, default)
#   4. Router to classify user input
#   5. Delegation to correct tool
#   6. Main async chat_agent function for FastAPI
# ------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI  # Wrapper for Google Gemini LLM
from langchain_core.prompts import ChatPromptTemplate        # Structured prompts for LLM
from langchain_core.output_parsers import StrOutputParser   # Convert model output to string
from langchain_core.runnables import RunnableBranch, RunnablePassthrough  # Routing system
from langchain.memory import ConversationBufferMemory       # In-memory chat history
from dotenv import load_dotenv                               # Load API keys from .env
import json
import os
import asyncio  # For running async LLM calls in routing

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()  # Loads API keys or config from a .env file

# ============================================================
#                     LLM (GOOGLE GEMINI MODEL)
# ============================================================
# Instantiate the Google Gemini LLM wrapper
# convert_system_message_to_human=True ensures system messages are understandable
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)

# ============================================================
#                      MEMORY SYSTEM
# ============================================================
# Stores past conversation for context, so model remembers previous messages
memory = ConversationBufferMemory(
    input_key="request",        # Key for user input
    memory_key="chat_history",  # Key where chat history is stored
    return_messages=True,       # Return full messages
    output_key="response"       # Key for model response
)

# ============================================================
#                   SAMPLE MARKS DATA
# ============================================================
# Static data for marks queries
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob": {"Math": 78, "Science": 85, "English": 80}
}

# ============================================================
#                         TOOLS
# ============================================================
# Each tool is an async function that sends a prompt to the LLM
# and returns a response. Tools are chosen based on user intent.

async def positive_tool(request, chat_history):
    """Handle happy messages."""
    prompt = f"You detected the user is happy.\nUser: {request}\nChat history: {chat_history}"
    res = await llm.ainvoke(prompt)  # async call to LLM
    return res.content

async def negative_tool(request, chat_history):
    """Handle sad or stressed messages."""
    prompt = f"User is sad or worried.\nUser: {request}\nChat history: {chat_history}"
    res = await llm.ainvoke(prompt)
    return res.content

async def marks_tool(request, chat_history):
    """Return student marks from predefined dictionary."""
    prompt = (
        f"Here is student marks data:\n{json.dumps(marks_data, indent=2)}\n\n"
        f"User question: {request}\nChat history: {chat_history}"
    )
    res = await llm.ainvoke(prompt)
    return res.content

async def suicide_tool_dynamic(request, chat_history):
    """Safe response for self-harm messages."""
    prompt = (
        f"User shows signs of self-harm.\n"
        f"Talk safely, calmly, and never give harmful instructions.\n"
        f"User: {request}\nChat history: {chat_history}"
    )
    res = await llm.ainvoke(prompt)
    return res.content

async def default_tool(request, chat_history):
    """Fallback tool for generic queries."""
    prompt = f"General query.\nUser: {request}\nChat history: {chat_history}"
    res = await llm.ainvoke(prompt)
    return res.content

# ============================================================
#                 ROUTER PROMPT (INTENT CLASSIFIER)
# ============================================================
# This LLM-based classifier categorizes user messages into:
# positive | negative | marks | suicide | default
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a classifier. Categorize the user request:

        positive → happiness
        negative → sadness/complaint
        marks → student marks query
        suicide → self-harm signals
        default → everything else

        Output ONLY: positive | negative | marks | suicide | default
    """),
    ("user", "{request}")  # Placeholder for user message
])

# Router chain = prompt → LLM → parse output as string
router_chain = router_prompt | llm | StrOutputParser()

# ============================================================
#                DECISION CHECK FUNCTIONS
# ============================================================
# These check the router's decision and return True/False
def is_positive(x): return x["decision"] == "positive"
def is_negative(x): return x["decision"] == "negative"
def is_marks(x): return x["decision"] == "marks"
def is_suicide(x): return x["decision"] == "suicide"

# ============================================================
#                      TOOL BRANCHES
# ============================================================
# Each branch runs the correct tool and labels which tool was used.
# asyncio.run() is used because RunnablePassthrough expects a synchronous lambda.

positive_branch = RunnablePassthrough().assign(
    text=lambda x: asyncio.run(positive_tool(x["request"], x["chat_history"])),
    tool=lambda x: "positive"
)

negative_branch = RunnablePassthrough().assign(
    text=lambda x: asyncio.run(negative_tool(x["request"], x["chat_history"])),
    tool=lambda x: "negative"
)

marks_branch = RunnablePassthrough().assign(
    text=lambda x: asyncio.run(marks_tool(x["request"], x["chat_history"])),
    tool=lambda x: "marks"
)

suicide_branch = RunnablePassthrough().assign(
    text=lambda x: asyncio.run(suicide_tool_dynamic(x["request"], x["chat_history"])),
    tool=lambda x: "suicide"
)

default_branch = RunnablePassthrough().assign(
    text=lambda x: asyncio.run(default_tool(x["request"], x["chat_history"])),
    tool=lambda x: "default"
)

# ============================================================
#                   ROUTER DECISION ENGINE
# ============================================================
# RunnableBranch checks each condition in order and runs the matching branch
delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch  # fallback if none match
)

# ============================================================
#                  MAIN CHAT AGENT FUNCTION
# ============================================================
async def chat_agent(user_input):
    """
    Main async chat agent used by FastAPI /chat endpoint.
    Steps:
      1. Load chat history from memory
      2. Classify user input (intent detection)
      3. Route to correct tool branch
      4. Save conversation in memory
      5. Return structured response
    """
    # Step 1: Load past chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Step 2: Classify intent using async router
    decision = (await router_chain.ainvoke({"request": user_input})).strip()

    # Step 3: Prepare data for delegation
    data = {
        "request": user_input,
        "chat_history": chat_history,
        "decision": decision
    }

    # Step 4: Run the correct tool branch
    result = await delegation_chain.arun(data)

    # Step 5: Save conversation to memory
    memory.save_context(
        {"request": user_input},
        {"response": result["text"], "tool_used": result["tool"]}
    )

    # Step 6: Return structured response
    return {"response": result["text"], "tool_used": result["tool"]}
