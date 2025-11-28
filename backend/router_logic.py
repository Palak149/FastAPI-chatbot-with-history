# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI      # Main LLM (Gemini)
from langchain_core.prompts import ChatPromptTemplate           # For structured prompt templates
from langchain_core.output_parsers import StrOutputParser       # Converts model output to clean string
from langchain.memory import ConversationBufferMemory           # Stores conversational history
from dotenv import load_dotenv                                  # Loads GOOGLE_API_KEY from environment
import json                                                     # Used for passing dicts as JSON text
import asyncio                                                  # Enables async tools


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
# .env must contain: GOOGLE_API_KEY="your_key_here"
load_dotenv()


# ---------------------------------------------------------
# Initialize Google Gemini LLM
# ---------------------------------------------------------
# convert_system_message_to_human=True:
#   Makes Gemini treat system prompts as if sent by a human,
#   improving adherence after an update in Gemini behavior.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)


# ---------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------
# This memory stores ALL chat history (user+AI messages)
# so the LLM can maintain context across multiple turns.
memory = ConversationBufferMemory(
    input_key="request",    # The name of the field where user input enters
    memory_key="chat_history",  # Where memory will be stored
    output_key="response",      # Name of stored LLM output
    return_messages=True         # Store messages as objects instead of plain text
)


# ---------------------------------------------------------
# Sample marks dataset (used by the marks tool)
# ---------------------------------------------------------
# This dictionary is NOT used to generate direct responses.
# Instead, the LLM receives this data and decides how to answer.
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob":   {"Math": 78, "Science": 85, "English": 80}
}


# ---------------------------------------------------------
# Async LLM-based TOOLS
# ---------------------------------------------------------
# Every tool builds a different prompt style for the LLM.
# There are absolutely NO predefined replies — the LLM
# generates the complete response every time.
# ---------------------------------------------------------

async def positive_tool(request, chat_history):
    """
    Used when user expresses happiness or positivity.
    Creates a positive, encouraging style response.
    """
    prompt = (
        f"The user sounds happy.\n\n"
        f"User: {request}\n\n"
        f"Chat history: {chat_history}\n\n"
        f"Reply in an uplifting, warm, positive tone."
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def negative_tool(request, chat_history):
    """
    Used when user is sad, stressed, angry, confused, or upset.
    The LLM replies empathetically — not predefined.
    """
    prompt = (
        f"The user seems sad, stressed, or upset.\n\n"
        f"User: {request}\n\n"
        f"Chat history: {chat_history}\n\n"
        f"Respond with emotional comfort and empathy."
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def marks_tool(request, chat_history):
    """
    When user asks about academic marks.
    The LLM gets the marks JSON and forms a natural-language answer.
    """
    prompt = (
        f"Here is the student marks dataset:\n"
        f"{json.dumps(marks_data, indent=2)}\n\n"
        f"User question: {request}\n"
        f"Chat history: {chat_history}\n\n"
        f"Use the dataset above to answer the question naturally."
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def suicide_tool_dynamic(request, chat_history):
    """
    SAFETY CRITICAL TOOL — used for suicidal or self-harm messages.
    NO fixed answer — the LLM generates empathetic, safe messages.
    """
    prompt = (
        "The user has expressed self-harm or suicidal thoughts.\n"
        "Your response MUST:\n"
        "- Be extremely empathetic\n"
        "- Encourage reaching out to close friends/family\n"
        "- Suggest contacting a professional or helpline\n"
        "- NOT provide any instructions for self-harm\n\n"
        f"User: {request}\n\n"
        f"Chat history: {chat_history}"
    )
    result = await llm.ainvoke(prompt)
    return result.content


async def default_tool(request, chat_history):
    """
    General conversation tool used when no specific category matches.
    """
    prompt = (
        f"General conversation.\n\n"
        f"User: {request}\n"
        f"Chat history: {chat_history}\n\n"
        f"Respond naturally and helpfully."
    )
    result = await llm.ainvoke(prompt)
    return result.content


# ---------------------------------------------------------
# ROUTER LOGIC — The LLM chooses the intent category
# ---------------------------------------------------------
# The LLM decides between:
#   positive | negative | marks | suicide | default
#
# The response of this router is a SINGLE WORD.
# ---------------------------------------------------------

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are an intent classifier.

        Classify the user's message into EXACTLY one:
        - positive
        - negative
        - marks
        - suicide
        - default

        Rules:
        positive → user is happy, joyful, excited
        negative → user is sad, stressed, angry, upset
        marks → user asks about academic marks
        suicide → self-harm or suicidal intent
        default → anything else

        Output ONLY one word.
    """),
    ("user", "{request}")
])

router_chain = router_prompt | llm | StrOutputParser()


# ---------------------------------------------------------
# MAIN CHAT AGENT (async)
# ---------------------------------------------------------
# Steps:
#   1. Load conversation history
#   2. Ask LLM to classify intent
#   3. Call correct tool
#   4. Save the new interaction into memory
#   5. Return LLM response + tool used
#
# FastAPI calls this function.
# ---------------------------------------------------------

async def chat_agent(user_input):
    """
    This is the main entry point used by FastAPI (/chat endpoint).

    Every response is generated by Gemini via LangChain.
    """

    # 1. Load memory/history (entire conversation so far)
    chat_history = memory.load_memory_variables({})["chat_history"]

    # 2. LLM classifies intent
    decision_raw = await router_chain.ainvoke({"request": user_input})
    decision = decision_raw.strip().lower()
    decision = decision.split()[0]  # Remove any accidental extra words

    # 3. Route request to the selected tool
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

    # 4. Save new conversation turn to memory
    memory.save_context(
        {"request": user_input},
        {"response": llm_response, "tool_used": tool_used}
    )

    # 5. Returned to FastAPI → sent to frontend
    return {
        "response": llm_response,  # LLM-generated message
        "tool_used": tool_used     # The tool category used
    }
