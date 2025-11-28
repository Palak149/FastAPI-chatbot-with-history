from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import json
import os

load_dotenv()

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)

# ---------------- MEMORY ----------------
memory = ConversationBufferMemory(
    input_key="request",
    memory_key="chat_history",
    return_messages=True,
    output_key="response"
)

# ---------------- Sample Marks Data ----------------
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob": {"Math": 78, "Science": 85, "English": 80}
}

# ---------------- TOOLS ----------------
def positive_tool(request, chat_history):
    prompt = f"You detected the user is happy.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

def negative_tool(request, chat_history):
    prompt = f"User is sad or worried.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content

def marks_tool(request, chat_history):
    prompt = (
        f"Here is student marks data:\n{json.dumps(marks_data, indent=2)}\n\n"
        f"User question: {request}\nChat history: {chat_history}"
    )
    return llm.invoke(prompt).content

def suicide_tool_dynamic(request, chat_history):
    prompt = (
        f"User shows signs of self-harm.\n"
        f"Talk safely, calmly, and never give harmful instructions.\n"
        f"User: {request}\nChat history: {chat_history}"
    )
    return llm.invoke(prompt).content

def default_tool(request, chat_history):
    prompt = f"General query.\nUser: {request}\nChat history: {chat_history}"
    return llm.invoke(prompt).content


# ---------------- ROUTER ----------------
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
    ("user", "{request}")
])

router_chain = router_prompt | llm | StrOutputParser()


# ---------------- DECISION CONDITIONS ----------------
def is_positive(x): return x["decision"] == "positive"
def is_negative(x): return x["decision"] == "negative"
def is_marks(x): return x["decision"] == "marks"
def is_suicide(x): return x["decision"] == "suicide"


# ---------------- CLEAN FIXED BRANCHES ----------------
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

# ---------------- ROUTER BRANCH ----------------
delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch
)


# ---------------- MAIN CHAT AGENT ----------------
def chat_agent(user_input):

    # Load chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Step 1: classify the request
    decision = router_chain.invoke({"request": user_input}).strip()

    # Construct data for routing
    data = {
        "request": user_input,
        "chat_history": chat_history,
        "decision": decision
    }

    # Step 2: route to correct tool
    result = delegation_chain.invoke(data)

    # Step 3: save memory
    memory.save_context(
        {"request": user_input},
        {"response": result["text"], "tool_used": result["tool"]}
    )

    return {
        "response": result["text"],
        "tool_used": result["tool"]
    }

