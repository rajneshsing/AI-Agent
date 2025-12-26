from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import streamlit as st

# -------------------------
# State definition (same as before)
# -------------------------
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# -------------------------
# Lazy LLM initialization (FIXED: safe for Streamlit Cloud)
# -------------------------
def get_llm():
    # Access secret only when function is called (not at import time)
    api_key = st.secrets["GROQ_API_KEY"]  # Will raise clear error if missing
    
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        model="llama-3.1-8b-instant",
        model_provider="groq",
        temperature=0.7,
        api_key=api_key
    )

# -------------------------
# Chat node (uses LLM only when needed)
# -------------------------
def chat_node(state: ChatState):
    llm = get_llm()  # Initialized here, safe
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# -------------------------
# SQLite checkpointer (simple, same as before)
# -------------------------
# Create connection once (Streamlit caches this automatically when used in graph)
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
memory = SqliteSaver(conn)

# -------------------------
# Build and compile the graph (very similar to original)
# -------------------------
builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

# Compile with memory
chatbot = builder.compile(checkpointer=memory)
