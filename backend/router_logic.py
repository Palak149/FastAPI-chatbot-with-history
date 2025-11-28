# =========================================================
# Router logic for Chatbot using LangChain + Google Gemini
# =========================================================
# Features:
# 1. Uses Google Gemini LLM as main brain
# 2. Routes user messages to the correct tool:
#       positive, negative, marks, suicide, default
# 3. Maintains chat history using LangChain ConversationBufferMemory
# 4. Returns a structured output: {"response": "...", "tool_used": "..."}
# =========================================================

# -------------------------
# Imports
# -------------------------
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini LLM
from langchain_core.prompts import ChatPromptTemplate       # For creating structured prompts
from langchain_core.output_parsers import StrOutputParser  # Cleans raw LLM output
from langchain_core.runnables import RunnableBranch, RunnablePassthrough  # Branching logic
from langchain.memory import ConversationBufferMemory       # Stores chat history
from dotenv import load_dotenv                               # Loads GOOGLE_API_KEY from .env
import json                                                 # For serializing marks dataset
import os

# -------------------------
# Load environment variables
# -------------------------
# Ensures GOOGLE_API_KEY is loaded from .env
load_dotenv()

# =========================================================
#  Initialize LLM (Google Gemini)
# ---------------------------------------------------------
# convert_system_message_to_human=True helps Gemini better
# understand system prompts. Note: this option may be deprecated.
# =========================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)

# =========================================================
#  Memory Setup
# ---------------------------------------------------------
# ConversationBufferMemory stores the context (chat history)
# Only stores user input ("request") and AI output ("response")
# =========================================================
memory = ConversationBufferMemory(
    input_key="request",        # Key used for user message
    memory_key="chat_history",  # Internal key for storing memory
    return_messages=True,       # Store messages as objects (not just text)
    output_key="response"       # Key for LLM reply
)

# =========================================================
#  Sample Marks Dataset
# ---------------------------------------------------------
# Used in marks_tool for responding to academic queries
# =========================================================
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob": {"Math": 78, "Science": 85, "English": 80}
}

# =========================================================
#  Tool Definitions
# ---------------------------------------------------------
# Each tool generates a specialized LLM prompt based on user input
# and returns the LLM response text.
# ---------------------------------------------------------
def positive_tool(request, chat_history):
    """Tool for positive/happy messages"""
    prompt = f"You detected the user is happy.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

def negative_tool(request, chat_history):
    """Tool for negative/sad/stressed messages"""
    prompt = f"User is sad or worried.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

def marks_tool(request, chat_history):
    """Tool for answering academic/marks queries"""
    prompt = (
        f"Here is student marks data:\n{json.dumps(marks_data, indent=2)}\n\n"
        f"User question: {request}\nChat history: {chat_history}"
    )
    return llm.invoke(prompt).content

def suicide_tool_dynamic(request, chat_history):
    """Tool for safe responses to suicidal/self-harm messages"""
    prompt = (
        f"User shows signs of self-harm.\n"
        f"Talk safely, calmly, and never give harmful instructions.\n"
        f"User: {request}\nChat history: {chat_history}"
    )
    return llm.invoke(prompt).content

def default_tool(request, chat_history):
    """General purpose tool for unclassified messages"""
    prompt = f"General query.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

# =========================================================
#  Router Prompt (Intent Classifier)
# ---------------------------------------------------------
# Uses LLM to classify user message into one of five categories:
# positive | negative | marks | suicide | default
# =========================================================
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
    ("user", "{request}")  # Placeholder for user input
])

# Chain the router with LLM and parser to clean output
router_chain = router_prompt | llm | StrOutputParser()

# =========================================================
#  Decision Conditions
# ---------------------------------------------------------
# Functions used by RunnableBranch to select which tool to execute
# =========================================================
def is_positive(x): return x["decision"] == "positive"
def is_negative(x): return x["decision"] == "negative"
def is_marks(x): return x["decision"] == "marks"
def is_suicide(x): return x["decision"] == "suicide"

# =========================================================
#  Branches (RunnablePassthrough)
# ---------------------------------------------------------
# Each branch executes its respective tool and assigns tool name
# =========================================================
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

# =========================================================
#  Router Branch
# ---------------------------------------------------------
# RunnableBranch decides which branch to execute based on conditions
# =========================================================
delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch  # Fallback if no condition matches
)

# =========================================================
#  Main Chat Agent
# ---------------------------------------------------------
# 1. Loads memory
# 2. Classifies input using router
# 3. Routes to correct tool branch
# 4. Stores user + LLM response in memory
# 5. Returns response + tool_used
# =========================================================
def chat_agent(user_input):

    # Load past chat history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Step 1: classify the user input
    decision = router_chain.invoke({"request": user_input}).strip()

    # Step 2: prepare data for routing
    data = {
        "request": user_input,
        "chat_history": chat_history,
        "decision": decision
    }

    # Step 3: route to the correct tool branch
    result = delegation_chain.invoke(data)

    # Step 4: store only user input + LLM response in memory
    # Note: tool_used is NOT stored in LangChain memory
    memory.save_context(
        {"request": user_input},
        {"response": result["text"], "tool_used": result["tool"]}
    )

    # Step 5: return response and tool used
    return {
        "response": result["text"],
        "tool_used": result["tool"]
    }
