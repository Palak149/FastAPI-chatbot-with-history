# ====================================================================
# router_logic.py
# ====================================================================
# This file handles the main chatbot logic for FastAPI:
# 1. Sets up the Google Gemini LLM via langchain_google_genai
# 2. Implements conversation memory using LangChain
# 3. Defines async tools for different intents (positive, negative, marks, suicide, default)
# 4. Uses a LangChain-based router to classify user input
# 5. Routes user messages to the correct tool asynchronously
# 6. Stores conversation in memory
# ====================================================================

from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini LLM wrapper
from langchain_core.prompts import ChatPromptTemplate        # Structured prompts
from langchain_core.output_parsers import StrOutputParser   # Parse LLM output to string
from langchain.memory import ConversationBufferMemory       # In-memory chat history
from dotenv import load_dotenv                               # Load API keys
import json
import asyncio  # For async LLM calls

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()  # Load API keys or config from a .env file

# ============================================================
#                     LLM (GOOGLE GEMINI)
# ============================================================
# Instantiate the LLM with convert_system_message_to_human=True
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)

# ============================================================
#                      MEMORY SYSTEM
# ============================================================
# Stores past conversation context so the model can reference it
memory = ConversationBufferMemory(
    input_key="request",        # User input key
    memory_key="chat_history",  # Key where chat history is stored
    return_messages=True,       # Return full chat history messages
    output_key="response"       # LLM output key
)

# ============================================================
#                   SAMPLE MARKS DATA
# ============================================================
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob": {"Math": 78, "Science": 85, "English": 80}
}

# ============================================================
#                         TOOLS
# ============================================================
# Each tool is an async function that sends a prompt to the LLM
# and returns the content. Tools are selected based on intent.

async def positive_tool(request, chat_history):
    """Handle happy/positive messages."""
    prompt = f"You detected the user is happy.\nUser: {request}\nChat history: {chat_history}"
    res = await llm.ainvoke(prompt)
    return res.content

async def negative_tool(request, chat_history):
    """Handle sad, worried, or stressed messages."""
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
    """Safe response tool for self-harm messages."""
    prompt = (
        f"User shows signs of self-harm.\n"
        f"Talk safely, calmly, and never give harmful instructions.\n"
        f"User: {request}\nChat history: {chat_history}"
    )
    res = await llm.ainvoke(prompt)
    return res.content

async def default_tool(request, chat_history):
    """Fallback tool for generic or uncategorized queries."""
    prompt = f"General query.\nUser: {request}\nChat history: {chat_history}"
    res = await llm.ainvoke(prompt)
    return res.content

# ============================================================
#                 ROUTER PROMPT (INTENT CLASSIFIER)
# ============================================================
# LLM-based classifier that categorizes user messages
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

# Chain: prompt -> LLM -> parse string
router_chain = router_prompt | llm | StrOutputParser()

# ============================================================
#                  MAIN CHAT AGENT FUNCTION
# ============================================================
async def chat_agent(user_input: str):
    """
    Async chat agent for FastAPI /chat endpoint.

    Steps:
    1. Load chat history from memory
    2. Classify user input (intent detection)
    3. Route to correct async tool
    4. Save conversation in memory
    5. Return structured response
    """
    # Step 1: Load chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Step 2: Classify intent
    decision = (await router_chain.ainvoke({"request": user_input})).strip()

    # Step 3: Route to the correct async tool
    if decision == "positive":
        text = await positive_tool(user_input, chat_history)
        tool_used = "positive"
    elif decision == "negative":
        text = await negative_tool(user_input, chat_history)
        tool_used = "negative"
    elif decision == "marks":
        text = await marks_tool(user_input, chat_history)
        tool_used = "marks"
    elif decision == "suicide":
        text = await suicide_tool_dynamic(user_input, chat_history)
        tool_used = "suicide"
    else:
        text = await default_tool(user_input, chat_history)
        tool_used = "default"

    # Step 4: Save conversation to memory
    memory.save_context(
        {"request": user_input},
        {"response": text, "tool_used": tool_used}
    )

    # Step 5: Return structured response
    return {"response": text, "tool_used": tool_used}
