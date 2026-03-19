\# 🔄 Self-Correcting RAG with LangGraph



\[!\[Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

\[!\[LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)

\[!\[Ollama](https://img.shields.io/badge/Ollama-Local-purple)](https://ollama.com)

\[!\[License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)



A production-ready \*\*Self-Correcting RAG\*\* system built with LangGraph that \*\*grades document relevance\*\* and \*\*rewrites queries\*\* when retrieval fails—just like a human would!



\## ✨ Features



\- \*\*🧠 Self-Correction\*\*: Automatically detects irrelevant documents and rewrites queries

\- \*\*📊 Document Grading\*\*: LLM-as-a-judge evaluates relevance of each retrieved chunk

\- \*\*🔄 Feedback Loop\*\*: Rewritten queries trigger fresh retrieval attempts (max 2 retries)

\- \*\*🏠 100% Local\*\*: Runs entirely on Ollama - no API costs, no rate limits

\- \*\*⚡ Optimized Performance\*\*: Vector store caching for 3-5x faster responses

\- \*\*🚀 Multiple Interfaces\*\*: CLI, Streamlit UI, and REST API ready



\## 🏗️ Architecture

User Question → RETRIEVE → GRADE → if relevant → GENERATE → Answer

↓

if NOT relevant

↓

REWRITE → RETRIEVE (max 2 attempts)



text



\## 📋 Prerequisites



\- Python 3.9+

\- \[Ollama](https://ollama.com) with models:

&#x20; ```bash

&#x20; ollama pull tinyllama

&#x20; ollama pull nomic-embed-text

🚀 Quick Start

bash

\# Clone the repository

git clone https://github.com/shankarlr/self-correcting-rag-langgraph.git

cd self-correcting-rag-langgraph



\# Create and activate virtual environment

python -m venv venv

\# On Windows:

venv\\Scripts\\activate

\# On Mac/Linux:

source venv/bin/activate



\# Install dependencies

pip install -r requirements.txt



\# Run the application

python crag\_demo.py "What is an AI agent?"  # CLI mode

streamlit run app.py                         # Web UI mode

📊 Performance

Query Type	First Run	Subsequent Runs

Cold Start	30-50s	-

Different Questions	-	20-40s

Same Question Repeated	-	1-3s

🛠️ Project Structure

text

self-correcting-rag-langgraph/

├── crag\_demo.py          # Core RAG implementation

├── app.py                # Streamlit UI

├── requirements.txt      # Python dependencies

├── .gitignore            # Git ignore rules

└── README.md             # This file

📝 License

MIT © Shankar L RF

