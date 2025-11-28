# =========================================================
#  ROUTER LOGIC FOR CHATBOT (LANGCHAIN + GEMINI)
# =========================================================
#  Features:
#   • Uses Google Gemini as the LLM
#   • Routes message to the correct tool (positive, negative, marks, etc.)
#   • Maintains chat history using LangChain memory
#   • Returns {"response": "...", "tool_used": "..."} to FastAPI
#   • Memory stores ONLY user and AI text (tool_used is NOT stored)
# =========================================================


# -------------------------
# Imports
# -------------------------
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini LLM
from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
from langchain_core.output_parsers import StrOutputParser  # To clean LLM text output
from langchain.memory import ConversationBufferMemory  # Stores chat history
from dotenv import load_dotenv  # Loads GOOGLE_API_KEY from .env file
import json  # For dataset encoding
import asyncio  # Enables async tasks


# -------------------------
# Load API keys from .env
# -------------------------
load_dotenv()   # Loads GOOGLE_API_KEY automatically


# =========================================================
#  INITIALIZE THE MAIN LLM (GOOGLE GEMINI)
# =========================================================
# convert_system_message_to_human=True makes Gemini more reliable with system prompts
# =========================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)


# =========================================================
#  CONVERSATION MEMORY
# =========================================================
# LangChain memory stores:
#   • user message ("request")
#   • AI message ("response")
#
# IMPORTANT:
#   Memory MUST NOT store tool_used (LangChain does not support extra fields)
# =========================================================
memory = ConversationBufferMemory(
    input_key="request",        # incoming message key
    memory_key="chat_history",  # internal variable name
    output_key="response",      # outgoing LLM response key
    return_messages=True        # store messages as objects
)


# =========================================================
# SAMPLE MARKS DATASET
# Used by marks_tool only
# =========================================================
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob":   {"Math": 78, "Science": 85, "English": 80}
}


# =========================================================
#  TOOL DEFINITIONS (All async, all use the LLM)
#  Each tool takes:
#      request      → user input
#      chat_history → stored memory
# =========================================================

async def positive_tool(request, chat_history):
    """Tool for cheerful/happy tone replies."""
    prompt = (
        f"The user sounds happy.\n\n"
        f"User: {request}\n"
        f"Chat history: {chat_history}\n\n"
        f"Reply in an uplifting, warm tone."
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def negative_tool(request, chat_history):
    """Tool for sad, angry, stressed tone replies."""
    prompt = (
        f"The user seems sad or upset.\n\n"
        f"User: {request}\n"
        f"Chat history: {chat_history}\n\n"
        f"Reply with empathy and emotional comfort."
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def marks_tool(request, chat_history):
    """Tool for answering marks or academic-related questions."""
    prompt = (
        f"Here is the student marks data:\n"
        f"{json.dumps(marks_data, indent=2)}\n\n"
        f"User question: {request}\n"
        f"Chat history: {chat_history}\n\n"
        f"Answer using the dataset above."
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def suicide_tool_dynamic(request, chat_history):
    """Tool for safe responses to suicidal/self-harm messages."""
    prompt = (
        "The user has expressed thoughts of self-harm.\n"
        "Your reply must:\n"
        "- Be supportive\n"
        "- Encourage reaching out to family/friends\n"
        "- Recommend contacting professionals or a helpline\n"
        "- NEVER encourage self-harm\n\n"
        f"User: {request}\n"
        f"Chat history: {chat_history}"
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def default_tool(request, chat_history):
    """Tool for general conversation."""
    prompt = (
        f"General conversation.\n\n"
        f"User: {request}\n"
        f"Chat history: {chat_history}\n\n"
        f"Respond naturally and helpfully."
    )
    result = await llm.ainvoke(prompt)
    return result.content


# =========================================================
#  ROUTER (Intent Classifier)
#  LLM decides which tool to use.
# =========================================================
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an intent classifier.

        Classify the user's message into EXACTLY one category:

        positive / negative / marks / suicide / default

        Output ONLY the category name.
    """),

    ("user", "{request}")
])

router_chain = router_prompt | llm | StrOutputParser()


# =========================================================
#  MAIN CHAT AGENT (called by FastAPI)
# =========================================================
# Steps:
#   1. Load memory
#   2. Ask router which tool to use
#   3. Run the selected tool
#   4. Save user + AI message (ONLY) to memory
#   5. Return result + tool_used
# =========================================================

async def chat_agent(user_input):
    """
    Main function. Called from FastAPI.
    Returns:
        {
            "response": "...",
            "tool_used": "positive | negative | marks | suicide | default"
        }
    """

    # 1. Load past chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # 2. Router chooses tool
    decision_raw = await router_chain.ainvoke({"request": user_input})
    decision = decision_raw.strip().lower().split()[0]

    # 3. Run correct tool
    if decision == "positive":
        llm_response = await positive_tool(user_input, chat_history)
        tool_used = "positive"

    elif decision == "negative":
        llm_response = await negative_tool(user_input, chat_history)
        tool_used = "negative"

    elif decision == "marks":
        llm_response = await marks_tool(user_input, chat_history)
        tool_used = "marks"

    elif decision == "suicide":
        llm_response = await suicide_tool_dynamic(user_input, chat_history)
        tool_used = "suicide"

    else:
        llm_response = await default_tool(user_input, chat_history)
        tool_used = "default"

    # 4. Store ONLY user_input + response in memory
    #    (do NOT store tool_used → breaks LangChain)
    memory.save_context(
        {"request": user_input},
        {"response": llm_response}
    )

    # 5. Return final result to FastAPI
    return {
        "response": llm_response,
        "tool_used": tool_used
    }
