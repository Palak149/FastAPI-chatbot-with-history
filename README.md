LLM Router System using LangChain + Gemini

This project implements a dynamic LLM router in Python using LangChain and Google Gemini (gemini-2.5-flash).
The system classifies each user message (positive, negative, marks, suicide, default) and routes it to the correct tool branch, generating safe and context-aware responses.

 Features

Automatic message classification (positive, negative, marks, suicide, default)

 Conversation Memory (LangChain ConversationBufferMemory)

 Tool-based routing using RunnableBranch

 Gemini LLM integrated via LangChain

 Dynamic prompts for each type of user emotion/request

 Example student marks database

 Safe response handling for self-harm messages


 Architecture Overview
User Message
      ↓
  Router LLM
 (classification)
      ↓
RunnableBranch Router
      ↓
Selected Tool Branch
      ↓
Final Response + Tool Used
      ↓
Memory Updated


Architecture Diagram 
                    ┌────────────────────┐
                    │    User Message    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     Router LLM     │
                    │ (Classifies input) │
                    └─────────┬──────────┘
                              │  decision = 
                              │  positive / negative /
                              │  marks / suicide / default
                              ▼
           ┌───────────────────────────────────────────┐
           │           RunnableBranch Router            │
           └─────────┬──────────────┬───────────┬──────┘
                     │              │           │
     decision=positive│    decision=negative     │
                     │              │           │
                     ▼              ▼           ▼
        ┌─────────────────┐   ┌──────────────────┐
        │ positive_branch │   │ negative_branch  │
        └────────┬────────┘   └─────────┬────────┘
                 │                      │
                 │                      │
                 ▼                      ▼
        (Runs positive_tool)    (Runs negative_tool)

                     ┌──────────────────────────────┐
 decision=marks  --> │          marks_branch        │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                           (Runs marks_tool)

                     ┌──────────────────────────────┐
 decision=suicide -->│        suicide_branch        │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                          (Runs suicide_tool)

                     ┌──────────────────────────────┐
 decision=default -->│       default_branch         │
                     └──────────────────────────────┘
                               │
                               ▼
                     (Runs default_tool)

                              ▼
                    ┌────────────────────┐
                    │ Final Response     │
                    │ + Tool Used        │
                    └────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Saved to Memory   │
                    └────────────────────┘

Project Files
app/
│── router.py          # Router logic + branches + tools
│── agent.py           # Main chat agent
│── data.py            # Marks database (sample)
│── README.md

 How Routing Works
1️⃣ Router Prompt

A classifier LLM receives the user text and outputs:

positive | negative | marks | suicide | default

2️⃣ RunnableBranch Routing

LangChain’s RunnableBranch behaves like IF–ELSE for LLM workflows:

delegation_chain = RunnableBranch(
    (is_positive, positive_branch),
    (is_negative, negative_branch),
    (is_marks, marks_branch),
    (is_suicide, suicide_branch),
    default_branch
)


Whichever condition returns True, that branch is executed.

 Tools / Branches

Each tool contains:

custom prompt

Gemini LLM invocation

tool name

Tool Name	When Triggered	Description
positive	user is happy	responds energetically
negative	sadness / complaint	supportive tone
marks	questions about student marks	returns marks info
suicide	self-harm signals	safe empathetic response
default	all other queries	general response
 Conversation Memory

Uses:

ConversationBufferMemory(
    input_key="request",
    memory_key="chat_history",
    return_messages=True,
    output_key="response"
)


This stores:

past user requests

LLM responses

tool used

and passes them to tools for context.

 Sample Marks Database
marks_data = {
    "Alice": {"Math": 95, "Science": 88, "English": 92},
    "Bob":   {"Math": 78, "Science": 85, "English": 80}
}

 How to Use
Run the chat system
from agent import chat_agent

print(chat_agent("I am happy today."))

Example Output
{
  "response": "That's amazing! I'm happy for you 😊",
  "tool_used": "positive"
}

 Example Sessions
💬 1. Positive Input

User:
I am very happy today!

Classifier Output:
positive

Tool Fired:
positive_branch

 2. Marks Query

User:
What are Alice’s marks in Science?

Classifier Output:
marks

Tool Fired:
marks_branch

3. Negative Input

User:
I feel so stressed today.

Classifier Output:
negative

Tool Fired:
negative_branch

💬4. Self-harm Message

User:
I don't want to live anymore.

Classifier Output:
suicide

Tool Fired:
suicide_branch

 Libraries Used

LangChain Core

LangChain Google GenAI

Google Gemini 2.5 Flash

Python 3.10+

dotenv

 Key LangChain Concepts Used
Concept	Why Used
ChatPromptTemplate	structured routing prompt
RunnableBranch	condition-based tool routing
RunnablePassthrough.assign	attach tool outputs
ConversationBufferMemory	maintain chat history
StrOutputParser	clean classifier output
 Why RunnableBranch? (In One Line)

Because it allows IF–ELSE style routing between multiple tools based on LLM-generated decisions.

 Conclusion

This project demonstrates how to build an LLM agent with:

intelligent routing

tool calling

memory

safe response handling

modular and clean architecture

Ready for:

chatbots

support agents

emotion-aware assistants

educational tools


