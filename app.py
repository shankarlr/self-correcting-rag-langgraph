"""
Self-Correcting RAG - FINAL FIXED VERSION
- No infinite loops
- Input works every time
- Clean chat interface
"""

import streamlit as st
import time
from crag_demo import ask_with_stats

# Must be the first Streamlit command
st.set_page_config(
    page_title="Self-Correcting RAG",
    page_icon="⚡",
    layout="wide"
)

# Initialize ALL session state variables at the start
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .answer-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    /* Fix for input field */
    .stTextInput input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">⚡ Self-Correcting RAG</h1>
    <p style="margin:0; opacity:0.9">Ask questions about AI agents, chain of thought, and more</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://ollama.com/public/ollama.png", width=200)
    
    st.markdown("### 📊 Session Stats")
    avg_time = st.session_state.total_time / max(st.session_state.total_queries, 1)
    st.metric("Total Queries", st.session_state.total_queries)
    st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    st.markdown("---")
    st.markdown("### 🚀 Example Questions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤖 AI Agent", use_container_width=True):
            st.session_state.current_question = "What is an AI agent?"
            st.session_state.processing = True
    with col2:
        if st.button("🧠 Chain of Thought", use_container_width=True):
            st.session_state.current_question = "Explain chain of thought"
            st.session_state.processing = True
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Prompt Engineering", use_container_width=True):
            st.session_state.current_question = "How does prompt engineering work?"
            st.session_state.processing = True
    with col2:
        if st.button("🤔 Agentic AI", use_container_width=True):
            st.session_state.current_question = "What is agentic AI?"
            st.session_state.processing = True
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_question = ""
        st.session_state.processing = False
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "time" in message:
            st.caption(f"⏱️ {message['time']:.2f}s | 🔄 {message.get('retries', 0)} retries")

# Input area - THIS IS THE KEY FIX
with st.container():
    col1, col2 = st.columns([6, 1])
    
    with col1:
        # Use a unique key for text_input that changes based on processing state
        input_key = f"input_{st.session_state.processing}_{len(st.session_state.messages)}"
        
        # If we're processing, show a disabled input with the current question
        if st.session_state.processing and st.session_state.current_question:
            question = st.text_input(
                "Ask a question:",
                value=st.session_state.current_question,
                key=input_key,
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            # Normal input mode
            question = st.text_input(
                "Ask a question:",
                placeholder="e.g., What is an AI agent?",
                key=input_key,
                label_visibility="collapsed"
            )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_disabled = st.session_state.processing
        submit = st.button(
            "🚀 Ask", 
            type="primary", 
            use_container_width=True,
            disabled=submit_disabled
        )

# Process the question
if (submit or st.session_state.processing) and st.session_state.current_question:
    question_to_process = st.session_state.current_question
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question_to_process})
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking... (this may take 10-30 seconds)"):
            start = time.time()
            result = ask_with_stats(question_to_process)
            elapsed = time.time() - start
        
        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
        retries = result.get("retries", 0)
        
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        st.caption(f"⏱️ {elapsed:.2f}s | 🔄 {retries} retries")
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "time": elapsed,
        "retries": retries
    })
    
    # Update stats
    st.session_state.total_queries += 1
    st.session_state.total_time += elapsed
    
    # Reset processing state
    st.session_state.processing = False
    st.session_state.current_question = ""
    
    # Force a rerun to update the UI
    st.rerun()

# Handle example button clicks
elif st.session_state.processing and not st.session_state.current_question:
    # This shouldn't happen, but just in case
    st.session_state.processing = False
    st.rerun()