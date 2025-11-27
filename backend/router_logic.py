from langchain_google_genai import ChatGoogleGenerativeAI   # LLM wrapper for Google Gemini models
from langchain_core.prompts import ChatPromptTemplate        # To create structured prompts
from langchain_core.output_parsers import StrOutputParser     # Extract string output from model response
from langchain_core.runnables import RunnableBranch, RunnablePassthrough  # For routing logic
from langchain.memory import ConversationBufferMemory         # Memory for chat history
from dotenv import load_dotenv                                # Load environment variables
import json
import os

# Load API keys from .env
load_dotenv()

# ============================================================
#                   LLM (GOOGLE GEMINI MODEL)
# ============================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",                 # LLM model to use
    convert_system_message_to_human=True      # Converts system prompts so Gemini can understand them
)

# ============================================================
#                       MEMORY SYSTEM
# ============================================================
# ConversationBufferMemory stores chat history so model remembers previous messages.
memory = ConversationBufferMemory(
    input_key="request",        # What user sends
    memory_key="chat_history",  # Where the chat history is stored
    return_messages=True,       # Memory returns full past messages
    output_key="response"       # Where assistant responses are stored
)

# ============================================================
#                   SAMPLE MARKS DATA (STATIC)
# ============================================================
# This is used by marks_tool to answer academic queries.
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob": {"Math": 78, "Science": 85, "English": 80}
}

# ============================================================
#                           TOOLS
# ============================================================
# Tools are small functions that respond depending on user intent.

def positive_tool(request, chat_history):
    """
    Handles positive/happy messages.
    """
    prompt = f"You detected the user is happy.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

def negative_tool(request, chat_history):
    """
    Handles sadness, tension, complaints.
    """
    prompt = f"User is sad or worried.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

def marks_tool(request, chat_history):
    """
    Returns academic marks from a predefined dictionary.
    """
    prompt = (
        f"Here is student marks data:\n{json.dumps(marks_data, indent=2)}\n\n"
        f"User question: {request}\nChat history: {chat_history}"
    )
    return llm.invoke(prompt).content

def suicide_tool_dynamic(request, chat_history):
    """
    Special safe-response tool for self-harm related messages.
    Model must always respond calmly and safely.
    """
    prompt = (
        f"User shows signs of self-harm.\n"
        f"Talk safely, calmly, and never give harmful instructions.\n"
        f"User: {request}\nChat history: {chat_history}"
    )
    return llm.invoke(prompt).content

def default_tool(request, chat_history):
    """
    Generic fallback tool for all normal queries.
    """
    prompt = f"General query.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

# ============================================================
#                 ROUTER PROMPT (INTENT CLASSIFIER)
# ============================================================
# This LLM classifies the user's message into one of the categories.

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
    ("user", "{request}")   # Passes the actual user message
])

# Router chain = Prompt → LLM → Convert output to plain text
router_chain = router_prompt | llm | StrOutputParser()

# ============================================================
#                DECISION CHECK FUNCTIONS
# ============================================================
# These act as conditions for RunnableBranch to decide which tool runs.

def is_positive(x): return x["decision"] == "positive"
def is_negative(x): return x["decision"] == "negative"
def is_marks(x): return x["decision"] == "marks"
def is_suicide(x): return x["decision"] == "suicide"

# ============================================================
#                      TOOL BRANCHES
# ============================================================
# RunnablePassthrough() lets us attach tool output and tool name.

positive_branch = RunnablePassthrough().assign(
    text=lambda x: positive_tool(x["request"], x["chat_history"]),
    tool=lambda x: "positive"
)

negative_branch = RunnablePassthrough().assign(
    text=lambda x: negative_tool(x["request"], x["chat_history"]),
    tool=lambda x: "negative"
)

marks_branch = RunnablePassthrough().assign(
    text=lambda x: marks_tool(x["request"], x["chat_history"]),
    tool=lambda x: "marks"
)

suicide_branch = RunnablePassthrough().assign(
    text=lambda x: suicide_tool_dynamic(x["request"], x["chat_history"]),
    tool=lambda x: "suicide"
)

default_branch = RunnablePassthrough().assign(
    text=lambda x: default_tool(x["request"], x["chat_history"]),
    tool=lambda x: "default"
)

# ============================================================
#                   ROUTER DECISION ENGINE
# ============================================================
# RunnableBranch decides which branch to run based on the decision.

delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch  # fallback branch
)

# ============================================================
#                 MAIN CHAT AGENT FUNCTION
# ============================================================
def chat_agent(user_input):
    """
    1. Load chat history
    2. Classify the user input (intent detection)
    3. Route to the correct tool
    4. Save conversation into memory
    5. Return formatted output back to FastAPI
    """

    # Load full chat history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Step 1: Use router to classify type of message
    decision = router_chain.invoke({"request": user_input}).strip()

    # Data passed to routing system
    data = {
        "request": user_input,
        "chat_history": chat_history,
        "decision": decision
    }

    # Step 2: Send the request to correct tool branch
    result = delegation_chain.invoke(data)

    # Step 3: Save conversation to memory
    memory.save_context(
        {"request": user_input},
        {"response": result["text"], "tool_used": result["tool"]}
    )

    # Step 4: Return final structured response
    return {
        "response": result["text"],
        "tool_used": result["tool"]
    }
