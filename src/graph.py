"""
LangGraph Agent — StateGraph with LLM-based routing (Agentic behavior).

Graph flow:
  START → router_node (LLM decides)
          ↓
      [Conditional]
      ↙           ↘
  retrieve         general_answer
      ↓                  ↓
  rag_answer             ↓
      ↘           ↙
      save_memory
          ↓
         END
"""

import os
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.vector_store import VectorStore
from src.memory import MemoryManager

load_dotenv()


# ------------------------------------------------------------------ #
#  State schema
# ------------------------------------------------------------------ #
class AgentState(TypedDict):
    query: str
    context: str
    response: str
    route: str  # "rag" or "general"
    memory_context: str
    session_id: str
    tool_used: str  # For UI feedback


# ------------------------------------------------------------------ #
#  Shared resources (cached)
# ------------------------------------------------------------------ #
_vector_store = None
_llm = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=600,
        )
    return _llm


# ------------------------------------------------------------------ #
#  Node functions
# ------------------------------------------------------------------ #

def router_node(state: AgentState) -> dict:
    """
    LLM decides whether to use RAG tool or General tool.
    """
    llm = get_llm()
    query = state["query"]
    
    # If the vector store is empty, force general
    vs = get_vector_store()
    if vs.collection.count() == 0:
        return {"route": "general", "tool_used": "General Knowledge"}

    system_prompt = """You are a routing agent. You decide which tool to use.
    
    Tools:
    1. "rag" - Use this if the user asks about specific documents, files, PDFs, "this context", "the uploaded file", or specific details that would be found in a user's uploaded document.
    2. "general" - Use this for greeting, general questions, coding help, math, or questions unrelated to any specific document.

    If unsure, default to "general".
    
    Respond ONLY with a JSON object: {"tool": "rag"} or {"tool": "general"}
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        content = response.content.strip()
        
        # Clean up code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
            
        decision = json.loads(content)
        route = decision.get("tool", "general")
    except Exception:
        route = "general"
        
    return {"route": route, "tool_used": "RAG Tool (Document Search)" if route == "rag" else "General Knowledge Tool"}


def retrieve_node(state: AgentState) -> dict:
    """Search ChromaDB for relevant documents."""
    vs = get_vector_store()
    results = vs.search(state["query"])

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # If nothing relevant found, fallback to empty context?
    # Actually, if we are here, the router *expected* docs.
    if not docs or all(d.strip() == "" for d in docs):
        return {"context": "NO_RELEVANT_DOCUMENTS_FOUND"}

    chunks = []
    for i, doc in enumerate(docs[:3]):  # top 3
        # If there's a contact_info chunk, prioritize it?
        # The vector store search handles relevance.
        page = metas[i].get("page", "?")
        source_type = metas[i].get("type", "text")
        label = f"[Page {page}]" if source_type != "contact_info" else "[Contact Info]"
        chunks.append(f"{label} {doc}")

    return {"context": "\n\n".join(chunks)}


def rag_answer_node(state: AgentState) -> dict:
    """Generate answer grounded in document context."""
    llm = get_llm()
    
    context = state["context"]
    if context == "NO_RELEVANT_DOCUMENTS_FOUND":
        # Fallback to general if retrieval failed despite routing
        return general_answer_node(state)

    system_prompt = f"""You are a document-grounded assistant.

{state.get('memory_context', '')}

Instructions:
- Answer ONLY using the document context below
- Cite sources using [Page X] where available
- If the answer is not in the context, say "I couldn't find that information in the document."

Document Context:
{context}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"]),
    ]
    result = llm.invoke(messages)
    return {"response": result.content}


def general_answer_node(state: AgentState) -> dict:
    """Generate a general knowledge answer."""
    llm = get_llm()

    system_prompt = f"""You are a helpful AI assistant.

{state.get('memory_context', '')}

Provide a clear, helpful answer using your general knowledge."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"]),
    ]
    result = llm.invoke(messages)
    return {"response": result.content}


# ------------------------------------------------------------------ #
#  Router (conditional edge)
# ------------------------------------------------------------------ #
def route_decision(state: AgentState) -> Literal["retrieve", "general_answer"]:
    if state.get("route") == "rag":
        return "retrieve"
    return "general_answer"


# ------------------------------------------------------------------ #
#  Build the graph
# ------------------------------------------------------------------ #
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rag_answer", rag_answer_node)
    graph.add_node("general_answer", general_answer_node)

    # Edges
    graph.set_entry_point("router")
    
    # Conditional edge from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "general_answer": "general_answer"
        }
    )
    
    graph.add_edge("retrieve", "rag_answer")
    graph.add_edge("rag_answer", END)
    graph.add_edge("general_answer", END)

    return graph.compile()


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #
_compiled_graph = None


def get_agent():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(query: str, memory_manager: MemoryManager) -> dict:
    """Run the LangGraph agent. Returns response AND which tool was used."""
    agent = get_agent()

    memory_context = memory_manager.get_full_context()

    result = agent.invoke({
        "query": query,
        "context": "",
        "response": "",
        "route": "",
        "memory_context": memory_context,
        "session_id": memory_manager.session_id,
        "tool_used": ""
    })

    response = result["response"]
    tool_used = result.get("tool_used", "General Knowledge")

    # Save to memory
    memory_manager.add_turn(query, response)

    return {"response": response, "tool_used": tool_used}
