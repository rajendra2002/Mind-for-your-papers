"""
RAG utility functions — used as node helpers by the LangGraph agent.
Kept minimal; the graph orchestration lives in graph.py.
"""

import os
from groq import Groq
from src.vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()


def get_groq_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def rag_answer(query: str, context: str, memory_context: str = "") -> str:
    """Answer using document context."""
    client = get_groq_client()

    prompt = f"""You are a document-grounded assistant.

{memory_context}

Instructions:
- Answer ONLY using the document context below
- Cite sources using [Page X]
- If the answer is not in the context, say so clearly

Document Context:
{context}

Question:
{query}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=600,
        messages=[
            {"role": "system", "content": "You answer strictly from documents."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def general_answer(query: str, memory_context: str = "") -> str:
    """Answer from general knowledge."""
    client = get_groq_client()

    system = f"You are a helpful AI assistant.\n\n{memory_context}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=600,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content.strip()
