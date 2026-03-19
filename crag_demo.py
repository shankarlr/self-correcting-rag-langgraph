"""
SELF-CORRECTING RAG WITH LANGGRAPH - FINAL PRODUCTION VERSION
Preserves core self-correction while being reasonably fast.
- Grades documents (core feature)
- Rewrites queries when needed (core feature)
- Self-corrects with max retries
"""

import os
import pickle
import time
import sys
import threading
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Annotated, Literal
import operator

# Suppress verbose logging
import logging
logging.getLogger().setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Chroma import
try:
    from langchain_chroma import Chroma
except ImportError:
    os.system("pip install -q langchain-chroma")
    from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# ============ CONFIGURATION ============
VECTORSTORE_PATH = "chroma_db"
DOCS_CACHE = "docs_cache.pkl"
MAX_RETRIES = 2                # Self-correction needs retries
TOP_K_RESULTS = 3               # Enough documents to grade
SILENT_MODE = True
MODEL_NAME = "tinyllama"        # Fast enough for self-correction
TIMEOUT_SECONDS = 60

# ============ CACHES ============
query_cache = {}
llm_cache = {}

# ============ STATE ============
class GraphState(TypedDict):
    question: str
    documents: Annotated[List[Document], operator.add]
    generation: str
    retries: int
    needs_web_search: bool

# ============ VECTOR STORE ============
def get_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        if not SILENT_MODE:
            print("📂 Loading existing vector store...")
        vector_store = Chroma(
            collection_name='rag-chroma-local',
            embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=VECTORSTORE_PATH
        )
        return vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    
    if not SILENT_MODE:
        print("🆕 First time setup - creating vector store...")
        print("📚 Loading documents...")
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ]
    
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    
    with open(DOCS_CACHE, 'wb') as f:
        pickle.dump(docs, f)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,          # Standard size for good context
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    if not SILENT_MODE:
        print(f"   Created {len(splits)} chunks")
    
    vector_store = Chroma.from_documents(
        documents=splits,
        collection_name='rag-chroma-local',
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=VECTORSTORE_PATH
    )
    
    return vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

retriever = get_vectorstore()

# ============ CACHED RETRIEVAL ============
def cached_retrieve(question: str):
    if question in query_cache:
        return query_cache[question]
    docs = retriever.invoke(question)
    query_cache[question] = docs
    return docs

# ============ NODE FUNCTIONS ============
def retrieve(state: GraphState) -> Dict:
    question = state["question"]
    documents = cached_retrieve(question)
    return {"documents": documents, "question": question, "retries": state.get("retries", 0)}

def grade_documents(state: GraphState) -> Dict:
    """Core self-correction: grade each document and decide if rewrite needed"""
    class RelevanceGrade(BaseModel):
        binary_score: Literal['yes', 'no'] = Field(description="Is document relevant? (yes/no)")
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1, num_predict=10)
    
    grader_prompt = PromptTemplate(
        template="""Document: {document}
Question: {question}
Relevant? Answer exactly 'yes' or 'no':""",
        input_variables=["document", "question"],
    )
    
    grader_chain = grader_prompt | llm | StrOutputParser()
    
    filtered_docs = []
    relevant_count = 0
    
    for doc in state["documents"]:
        doc_preview = doc.page_content[:200]
        cache_key = f"grade_{hash(doc_preview)}_{state['question']}"
        
        if cache_key in llm_cache:
            grade_text = llm_cache[cache_key]
        else:
            grade_text = grader_chain.invoke({
                "document": doc_preview,
                "question": state["question"]
            })
            llm_cache[cache_key] = grade_text
        
        # Simple string parsing (more reliable than structured output)
        is_relevant = 'yes' in grade_text.lower().strip()
        
        if is_relevant:
            filtered_docs.append(doc)
            relevant_count += 1
    
    # Self-correction trigger: if no relevant docs, needs rewrite
    needs_search = (relevant_count == 0)
    
    return {
        "documents": filtered_docs,
        "needs_web_search": needs_search,
        "question": state["question"],
        "retries": state.get("retries", 0) + 1
    }

def rewrite_query(state: GraphState) -> Dict:
    """Rewrite query when no relevant documents found"""
    llm = ChatOllama(model=MODEL_NAME, temperature=0.7, num_predict=50)
    
    rewrite_prompt = PromptTemplate(
        template="""Original query: {question}
Improved query (more specific):""",
        input_variables=["question"],
    )
    
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    cache_key = f"rewrite_{state['question']}"
    if cache_key in llm_cache:
        better_question = llm_cache[cache_key]
    else:
        better_question = rewrite_chain.invoke({"question": state["question"]})
        llm_cache[cache_key] = better_question
    
    return {
        "question": better_question,
        "documents": [],
        "needs_web_search": False,
        "retries": state["retries"]
    }

def generate(state: GraphState) -> Dict:
    """Generate answer from relevant documents"""
    if not state["documents"]:
        context = "No relevant documents found."
    else:
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0.3, num_predict=200)
    
    generate_prompt = PromptTemplate(
        template="""Context: {context}
Question: {question}
Answer based only on context:""",
        input_variables=["context", "question"],
    )
    
    generate_chain = generate_prompt | llm | StrOutputParser()
    
    cache_key = f"gen_{hash(context[:200])}_{state['question']}"
    if cache_key in llm_cache:
        answer = llm_cache[cache_key]
    else:
        answer = generate_chain.invoke({
            "context": context,
            "question": state["question"]
        })
        llm_cache[cache_key] = answer
    
    return {"generation": answer, "question": state["question"], "documents": state["documents"]}

# ============ BUILD GRAPH ============
def decide_next_node(state: GraphState) -> str:
    """Self-correction decision logic"""
    if state["retries"] >= MAX_RETRIES:
        return "generate"
    if state.get("needs_web_search", False) and len(state.get("documents", [])) == 0:
        return "rewrite_query"
    return "generate"

def build_self_correcting_rag():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_node,
        {"rewrite_query": "rewrite_query", "generate": "generate"}
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile()

app = build_self_correcting_rag()

# ============ WARMUP ============
def warm_up():
    """Pre-load models without breaking self-correction"""
    try:
        _ = retriever.invoke("warm up")
        warmup_llm = ChatOllama(model=MODEL_NAME, temperature=0)
        _ = warmup_llm.invoke("Hello")
    except:
        pass

warmup_thread = threading.Thread(target=warm_up)
warmup_thread.daemon = True
warmup_thread.start()

# ============ API FUNCTIONS ============
def ask_with_stats(question: str) -> Dict:
    start = time.time()
    
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retries": 0,
        "needs_web_search": False
    }
    
    cache_key = f"full_{question}"
    if cache_key in llm_cache:
        result = llm_cache[cache_key]
        result["time"] = time.time() - start
        result["cached"] = True
        return result
    
    try:
        final_state = app.invoke(initial_state)
        elapsed = time.time() - start
        result = {
            "answer": final_state.get("generation", "No answer"),
            "retries": final_state.get("retries", 0),
            "time": elapsed,
            "success": True
        }
        llm_cache[cache_key] = result
        return result
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "retries": 0,
            "time": time.time() - start,
            "success": False
        }

def ask(question: str) -> str:
    return ask_with_stats(question).get("answer", "")

# ============ COMMAND LINE ============
if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = ask_with_stats(question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"\n⏱️  {result['time']:.2f}s | 🔄 {result['retries']} retries")
    else:
        print("\n" + "🔥"*35)
        print("   SELF-CORRECTING RAG")
        print("🔥"*35)
        print("\nType 'quit' to exit\n")
        
        while True:
            q = input("\n❓ Question: ")
            if q.lower() in ['quit', 'exit', 'q']:
                break
            result = ask_with_stats(q)
            print(f"\n✅ Answer: {result['answer'][:200]}..." if len(result['answer']) > 200 else f"\n✅ Answer: {result['answer']}")
            print(f"⏱️  {result['time']:.2f}s | 🔄 {result['retries']} retries")