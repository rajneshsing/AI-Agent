from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

# ✅ Correct env var for Groq
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -------------------------
# State definition
# -------------------------
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# -------------------------
# Initialize Groq LLM (FIXED)
# -------------------------
llm = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",   # ✅ MUST be model_provider
    temperature=0.7
)

# -------------------------
# Chat node
# -------------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# -------------------------
# Checkpointer (SQLite)
# -------------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------------
# Build graph
# -------------------------
builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

chatbot = builder.compile(checkpointer=checkpointer)
