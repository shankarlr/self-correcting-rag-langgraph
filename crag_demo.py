"""
SELF-CORRECTING RAG WITH LANGGRAPH - FINAL VERSION
Key Features:
- Self-correcting retrieval with document grading
- Local LLMs (Ollama) - no API costs
- Smart rewrite logic (only when needed)
- Prevents infinite loops with max retries
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# ============ PART 1: DEFINE STATE (Agent's Memory) ============

class GraphState(TypedDict):
    """State schema for our LangGraph agent"""
    question: str                          # Current question being processed
    documents: Annotated[List[Document], operator.add]  # Retrieved documents
    generation: str                         # Final generated answer
    retries: int                            # Number of rewrite attempts
    needs_web_search: bool                   # Flag for triggering rewrite
    
print("✅ State defined - agent memory structure ready")

# ============ PART 2: SETUP KNOWLEDGE BASE ============

def setup_vectorstore():
    """Load documents and create searchable vector database"""
    print("📚 Loading documents...")
    
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ]
    
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"   Created {len(splits)} document chunks")
    
    # Create vector store with local embeddings
    vector_store = Chroma.from_documents(
        documents=splits,
        collection_name='rag-chroma-local',
        embedding=OllamaEmbeddings(model="nomic-embed-text")  # Local embeddings!
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 4})  # Get 4 docs per query

retriever = setup_vectorstore()
print("✅ Vectorstore ready with local embeddings")

# ============ PART 3: NODE FUNCTIONS ============

def retrieve(state: GraphState) -> Dict:
    """
    Node 1: Retrieve relevant documents from vector database
    FIXED: Using .invoke() instead of deprecated ._get_relevant_documents()
    """
    print("\n---🔍 NODE: RETRIEVE ---")
    question = state["question"]
    
    # CORRECT: Use invoke() method (new LangChain style)
    documents = retriever.invoke(question)
    print(f"   Retrieved {len(documents)} documents")
    
    return {
        "documents": documents,
        "question": question,
        "retries": state.get("retries", 0)
    }
    
def grade_documents(state: GraphState) -> Dict:
    """
    Node 2: Grade document relevance - THE HEART OF SELF-CORRECTION
    FIXED: Only triggers rewrite if NO documents are relevant
    FIXED: Proper type hints (GraphState not StateGraph)
    """
    print("\n---⭐ NODE: GRADE DOCUMENTS ---")
    
    class RelevanceGrade(BaseModel):
        """Structured output for relevance grading"""
        binary_score: Literal['yes', 'no'] = Field(
            description="Document is relevant to question (yes/no)"
        )
    
    # Use local LLM for grading (low temperature for consistency)
    llm = ChatOllama(model="mistral", temperature=0)
    
    grader_prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        
        Retrieved document: {document}
        User question: {question}
        
        If the document contains keywords or meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no'.
        """,
        input_variables=["document", "question"],
    )   
    
    # Chain: prompt → LLM with structured output
    grader_chain = grader_prompt | llm.with_structured_output(RelevanceGrade)
    
    filtered_docs = []
    relevant_count = 0
    
    # Grade each document
    for i, doc in enumerate(state["documents"], 1):
        print(f"   Grading document {i}...")
        
        grade = grader_chain.invoke({
            "document": doc.page_content[:500],  # First 500 chars is enough
            "question": state["question"]
        })
        
        if grade.binary_score == "yes":
            print(f"      ✅ Document {i} - RELEVANT (keeping)")
            filtered_docs.append(doc)
            relevant_count += 1
        else:
            print(f"      ❌ Document {i} - NOT RELEVANT")
    
    # KEY FIX: Only trigger rewrite if NO documents are relevant
    # This prevents over-correction and infinite loops
    needs_search = (relevant_count == 0)
    
    if needs_search:
        print(f"   ⚠️ No relevant documents found (0/{len(state['documents'])}). Need rewrite.")
    else:
        print(f"   ✅ Found {relevant_count}/{len(state['documents'])} relevant documents. Good to generate.")
            
    return {
        "documents": filtered_docs,
        "needs_web_search": needs_search,
        "question": state["question"],
        "retries": state.get("retries", 0) + 1
    }
    
def rewrite_query(state: GraphState) -> Dict:
    """
    Node 3: Rewrite query when retrieval fails
    FIXED: Using correct Ollama model (mistral, not gpt-3.5-turbo)
    FIXED: Proper return structure
    """
    print("\n---✏️ NODE: REWRITE QUERY ---")
    
    # FIXED: Use mistral (Ollama model) with higher temperature for creativity
    llm = ChatOllama(model="mistral", temperature=0.7)
    
    rewrite_prompt = PromptTemplate(
        template="""You are generating an improved search query.
        Original question: {question}
        
        This query failed to retrieve relevant documents.
        Generate a better version that's more specific and uses key terms.
        
        Better query:""",
        input_variables=["question"],
    )
    
    # CORRECT chain order: prompt | llm | parser
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    better_question = rewrite_chain.invoke({"question": state["question"]})
    
    print(f"   Original: {state['question']}")
    print(f"   Rewritten: {better_question}")
    
    return {
        "question": better_question,
        "documents": [],  # Clear old documents
        "needs_web_search": False,  # Reset flag
        "retries": state["retries"]  # Keep retry count
    }
    
def generate(state: GraphState) -> Dict:
    """
    Node 4: Generate final answer from relevant documents
    FIXED: Correct chain order (prompt | llm | parser)
    FIXED: Proper prompt template with context
    """
    print("\n---🤖 NODE: GENERATE ---")
    
    if not state["documents"]:
        print("   ⚠️ No documents available - generating without context")
        context = "No relevant documents found. Answer based on general knowledge."
    else:
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    llm = ChatOllama(model="mistral", temperature=0)
    
    # FIXED: Correct template with context
    generate_prompt = PromptTemplate(
        template="""Answer the question based on the following context:
        
        Context: {context}
        
        Question: {question}
        
        Answer concisely and accurately based ONLY on the context provided:""",
        input_variables=["context", "question"],
    )
    
    # FIXED: Correct chain order (prompt | llm | parser)
    generate_chain = generate_prompt | llm | StrOutputParser()
    
    print(f"   Context length: {len(context)} chars")
    
    answer = generate_chain.invoke({
        "context": context,
        "question": state["question"]
    })
    
    print(f"   ✅ Generated answer of length {len(answer)} chars")
    
    return {
        "generation": answer,
        "question": state["question"],
        "documents": state["documents"]
    }
    
# ============ PART 4: BUILD THE GRAPH ============

def decide_next_node(state: GraphState) -> str:
    """
    Conditional edge - decides where to go next
    FIXED: Added max retries protection (2 attempts max)
    FIXED: Better logging for debugging
    """
    print("\n---🔀 DECIDING NEXT NODE ---")
    print(f"   Retries: {state['retries']}")
    print(f"   Needs web search: {state.get('needs_web_search', False)}")
    print(f"   Documents found: {len(state.get('documents', []))}")
    
    # FIXED: Max 2 retries to prevent infinite loops
    if state["retries"] >= 2:
        print("   ⚠️ MAX RETRIES REACHED - forcing generate")
        return "generate"
    
    # Only rewrite if we need search AND have no relevant documents
    if state.get("needs_web_search", False) and len(state.get("documents", [])) == 0:
        print("   ➡️ No relevant docs found - rewriting query")
        return "rewrite_query"
    else:
        print("   ➡️ Have relevant documents - generating answer")
        return "generate"

def build_self_correcting_rag():
    """
    Assemble all nodes into a LangGraph workflow
    FIXED: Using START constant (modern LangGraph)
    FIXED: Added return statement for compiled graph
    """
    # CORRECT: Pass GraphState schema to StateGraph
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    
    # Define the flow
    workflow.add_edge(START, "retrieve")  # Modern way: use START constant
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional edge from grade_documents
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_node,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate"
        }
    )  
    
    # Connect rewrite back to retrieve (the feedback loop!)
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    
    # FIXED: Return the compiled graph
    return workflow.compile()
    
# Build the graph
app = build_self_correcting_rag()
print("✅ Graph built and compiled - ready to run!")

# ============ PART 5: RUN THE AGENT ============

def run_rag_agent(question: str):
    """Run the agent with a question and show the process"""
    print("\n" + "="*70)
    print(f"🚀 QUESTION: {question}")
    print("="*70)
    
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retries": 0,
        "needs_web_search": False
    }
    
    try:
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*70)
        print("✅ FINAL ANSWER:")
        print("-"*70)
        print(final_state.get("generation", "No answer generated"))
        print("="*70)
        
        # Show stats
        if final_state.get("retries", 0) > 0:
            print(f"\n📊 Stats: Needed {final_state['retries']} rewrite attempt(s)")
        
        return final_state
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run tests
if __name__ == "__main__":
    print("\n" + "🔥"*35)
    print("   SELF-CORRECTING RAG WITH LOCAL LLMS")
    print("🔥"*35)
    
    # Test questions
    questions = [
        "What is an AI agent?",
        "Tell me about chain of thought",
        "How does prompt engineering work?",
        "What is machine learning?",  # Might not be in docs
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n--- TEST {i}/{len(questions)} ---")
        run_rag_agent(q)
        if i < len(questions):
            input("\nPress Enter for next question...")
    
    print("\n🎉 All tests complete!")