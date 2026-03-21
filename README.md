# PaperMind - AI-Powered Document Intelligence

> Upload a PDF, ask anything — get instant intelligent answers powered by AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange.svg)
![Groq](https://img.shields.io/badge/Groq-API-purple.svg)

---

## What is PaperMind?

PaperMind is an AI-powered chatbot that reads your PDF documents and answers questions about them in a beautiful ChatGPT-style interface. It can also answer general knowledge questions just like ChatGPT.

---

## Key Features

- Upload PDF documents and extract text, images, and contact info automatically
- Chat with your documents and get instant answers
- Smart routing — AI decides whether to search the document or use general knowledge
- Remembers last 10 messages in the current session
- Extracts and saves important facts across all sessions
- Describes images found inside PDFs
- Dark ChatGPT-style interface built with Streamlit

---

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Streamlit | Web interface |
| LangGraph | AI workflow and routing |
| Groq API | Fast LLM inference |
| ChromaDB | Vector database |
| PyMuPDF | PDF processing |
| Sentence Transformers | Text embeddings |
| LangChain | LLM framework |

---

## AI Models Used

| Model | Purpose |
|-------|---------|
| llama-3.1-8b-instant | Routing decisions |
| llama-3.2-11b-vision-preview | Image description |
| llama-3.3-70b-versatile | Answer generation |

---

## Project Structure

```
PaperMind/
│
├── src/
│   ├── app.py              # Streamlit web interface
│   ├── graph.py            # LangGraph AI routing engine
│   ├── ingest.py           # PDF processor
│   ├── vector_store.py     # ChromaDB vector store
│   ├── rag.py              # RAG answer generation
│   ├── memory.py           # Short and long-term memory
│   └── __init__.py
│
├── chat_sessions/          # Saved conversation history
├── chroma_db/              # Vector database storage
├── .env                    # API keys (not uploaded)
├── requirements.txt        # Python dependencies
├── diagnose.py             # Dependency checker
├── debug_env.py            # Environment debugger
└── run.bat                 # Windows launcher
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/rajendra2002/Mind-for-your-papers.git
cd Mind-for-your-papers
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at https://console.groq.com

### 5. Run the app
```bash
streamlit run src/app.py
```

The app will open automatically at http://localhost:8501

---

## How It Works

```
User uploads PDF
      ↓
ingest.py extracts text, images, contact info
      ↓
vector_store.py stores chunks in ChromaDB
      ↓
User asks a question
      ↓
graph.py routes the query:
   ├── Document question → Search ChromaDB → RAG answer
   └── General question → General AI answer
      ↓
memory.py saves conversation
      ↓
Answer displayed in UI
```

---

## Memory System

- Short-term memory keeps the last 10 conversation turns in the current session
- Long-term memory extracts key facts using AI and saves them to global_facts.json, shared across all sessions

---

## Troubleshooting

Run the diagnostic scripts if you face any issues:
```bash
python diagnose.py
python debug_env.py
```

---

## License

This project is open source and available under the MIT License.

---

## Author

Made by [Rajendra Choudhary](https://github.com/rajendra2002)
