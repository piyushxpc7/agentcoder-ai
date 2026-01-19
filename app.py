"""AgentCoder AI - Main Streamlit Application."""

import os
import sys
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.schemas import (
    CodeGenerationRequest, CodeAnalysisRequest, Language, StepType
)
from agents import CodeGenerationAgent, CodeAnalysisAgent, RAGAgent
from tools import PDFParser, CodeTools, RetrievalTools
from utils import VectorStore, DocumentStore, EmbeddingGenerator

load_dotenv()

# ============== Page Configuration ==============
st.set_page_config(
    page_title="AgentCoder AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============== Custom CSS ==============
st.markdown("""
<style>
    /* Dark theme with gradient accents */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Card styling */
    .agent-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Step indicators */
    .step-thought { border-left: 4px solid #3b82f6; padding-left: 12px; }
    .step-action { border-left: 4px solid #10b981; padding-left: 12px; }
    .step-observation { border-left: 4px solid #f59e0b; padding-left: 12px; }
    .step-final { border-left: 4px solid #8b5cf6; padding-left: 12px; }
    
    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 1.25rem;
    }
    .score-high { background: linear-gradient(135deg, #10b981, #059669); color: white; }
    .score-medium { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
    .score-low { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    /* Sidebar */
    .css-1d391kg { background: rgba(0,0,0,0.3); }
    
    /* File uploader */
    .uploadedFile { background: rgba(102, 126, 234, 0.1); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ============== Session State Initialization ==============
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    if "document_store" not in st.session_state:
        st.session_state.document_store = DocumentStore()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = os.getenv("DEFAULT_MODEL", "mistralai/mistral-small-3.1-24b-instruct:free")

init_session_state()


# ============== Sidebar ==============
def render_sidebar():
    """Render the sidebar with settings and document upload."""
    with st.sidebar:
        st.markdown('<h1 class="main-header">🤖 AgentCoder AI</h1>', unsafe_allow_html=True)
        st.markdown("*Intelligent Code Generation & Analysis*")
        st.divider()
        
        # Model Settings
        st.subheader("⚙️ Settings")
        models = [
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "mistralai/mistral-small-3.2-24b-instruct:free",
            "meta-llama/llama-3-8b-instruct:free",
        ]
        st.session_state.selected_model = st.selectbox(
            "Model", models, index=0
        )
        st.caption("🆓 All models are free! Some may have rate limits - try another if one is busy.")
        
        st.divider()
        
        # Document Upload
        st.subheader("📄 Documentation")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Upload documentation PDFs for RAG"
        )
        
        if uploaded_file:
            if st.button("📥 Index Document", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    process_uploaded_pdf(uploaded_file)
        
        # Document List
        docs = st.session_state.document_store.list_documents()
        if docs:
            st.markdown("**Indexed Documents:**")
            for doc_id, meta in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📎 {meta.filename[:20]}...")
                with col2:
                    if st.button("🗑️", key=f"del_{doc_id}"):
                        delete_document(doc_id)
                        st.rerun()
        
        st.divider()
        
        # Stats
        st.subheader("📊 Stats")
        st.metric("Documents", len(docs))
        st.metric("Total Chunks", st.session_state.vector_store.total_chunks)


def process_uploaded_pdf(uploaded_file):
    """Process and index an uploaded PDF file."""
    try:
        doc_id = str(uuid.uuid4())
        content = uploaded_file.read()
        
        # Parse PDF
        parser = PDFParser()
        chunks, page_count = parser.parse_pdf_bytes(content, doc_id)
        
        # Store document
        st.session_state.document_store.add_document(
            filename=uploaded_file.name,
            content=content,
            page_count=page_count,
            chunk_count=len(chunks)
        )
        
        # Index chunks
        st.session_state.vector_store.add_chunks(chunks)
        
        st.success(f"✅ Indexed {len(chunks)} chunks from {page_count} pages")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")


def delete_document(doc_id: str):
    """Delete a document and its chunks."""
    st.session_state.vector_store.delete_document(doc_id)
    st.session_state.document_store.delete_document(doc_id)


# ============== Agent Steps Display ==============
def render_agent_steps(steps):
    """Render agent reasoning steps."""
    if not steps:
        return
    
    with st.expander("🧠 Agent Reasoning", expanded=True):
        for step in steps:
            icon = {"thought": "💭", "action": "⚡", "observation": "👁️", "final_answer": "✅"}
            css_class = f"step-{step.step_type.value.replace('_', '-')}"
            
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{icon.get(step.step_type.value, '📌')} {step.step_type.value.replace('_', ' ').title()}</strong><br/>
                {step.content}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")


# ============== Tab 1: Code Generation ==============
def render_code_generation_tab():
    """Render the code generation tab."""
    st.markdown("### 💻 Code Generation")
    st.markdown("Describe what you want to build, and I'll generate the code.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Your Requirements",
            placeholder="E.g., Create a Python function that fetches weather data from an API...",
            height=150
        )
    
    with col2:
        language = st.selectbox(
            "Language",
            ["Auto-detect", "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust"],
        )
        include_tests = st.checkbox("Include Tests")
        include_docs = st.checkbox("Include Docs", value=True)
    
    if st.button("🚀 Generate Code", use_container_width=True):
        if not prompt:
            st.warning("Please enter your requirements")
            return
        
        with st.spinner("Generating code..."):
            try:
                agent = CodeGenerationAgent(st.session_state.selected_model)
                request = CodeGenerationRequest(
                    prompt=prompt,
                    language=Language[language.upper().replace("-", "_")] if language != "Auto-detect" else None,
                    include_tests=include_tests,
                    include_docs=include_docs,
                )
                response, steps = agent.generate_sync(request)
                st.session_state.agent_steps = steps
                
                # Display result
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 📝 {response.filename or 'Generated Code'}")
                    st.code(response.code, language=CodeTools.get_syntax_name(response.language))
                
                with col2:
                    st.markdown("**Language:** " + response.language.value.title())
                    if response.dependencies:
                        st.markdown("**Dependencies:**")
                        for dep in response.dependencies:
                            st.markdown(f"- `{dep}`")
                    
                    st.download_button(
                        "💾 Download",
                        response.code,
                        file_name=response.filename or "code.txt",
                        mime="text/plain"
                    )
                
                # Explanation
                with st.expander("📖 Explanation"):
                    st.markdown(response.explanation)
                
                if response.usage_example:
                    with st.expander("💡 Usage Example"):
                        st.code(response.usage_example)
                
                if response.tests:
                    with st.expander("🧪 Tests"):
                        st.code(response.tests, language=CodeTools.get_syntax_name(response.language))
                
                # Agent steps
                render_agent_steps(steps)
                
            except Exception as e:
                error_msg = str(e)
                # Handle rate limit errors
                if "429" in error_msg or "rate" in error_msg.lower() or "rate-limited" in error_msg.lower():
                    st.error("⚠️ **Rate Limit Error**: This model is temporarily rate-limited.")
                    st.info("💡 **Suggestion**: Try switching to a different model in the sidebar, or wait a few minutes and try again. Free models are more likely to be rate-limited.")
                else:
                    st.error(f"Generation failed: {error_msg}")


# ============== Tab 2: Code Analysis ==============
def render_code_analysis_tab():
    """Render the code analysis tab."""
    st.markdown("### 🔍 Code Analysis")
    st.markdown("Paste your code for a comprehensive review.")
    
    code_input = st.text_area(
        "Your Code",
        placeholder="Paste your code here...",
        height=300
    )
    
    focus_areas = st.multiselect(
        "Focus Areas",
        ["security", "performance", "best_practices", "readability", "error_handling"],
        default=["security", "performance", "best_practices"]
    )
    
    if st.button("🔬 Analyze Code", use_container_width=True):
        if not code_input:
            st.warning("Please paste your code")
            return
        
        with st.spinner("Analyzing code..."):
            try:
                agent = CodeAnalysisAgent(st.session_state.selected_model)
                request = CodeAnalysisRequest(code=code_input, focus_areas=focus_areas)
                result, steps = agent.analyze_sync(request)
                st.session_state.agent_steps = steps
                
                # Score display
                st.markdown("---")
                score_class = "score-high" if result.quality_score >= 70 else "score-medium" if result.quality_score >= 40 else "score-low"
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <span class="score-badge {score_class}">{result.quality_score}/100</span>
                    <p style="margin-top: 1rem; color: #94a3b8;">{result.summary}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ Strengths")
                    for s in result.strengths:
                        st.markdown(f"- {s}")
                
                with col2:
                    st.markdown("#### 📋 Recommendations")
                    for r in result.recommendations:
                        st.markdown(f"- {r}")
                
                # Issues
                if result.issues:
                    st.markdown("#### ⚠️ Issues Found")
                    for issue in result.issues:
                        color = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}
                        st.markdown(f"{color.get(issue.severity.value, '⚪')} **{issue.category}** (Line {issue.line or '?'}): {issue.message}")
                        if issue.suggestion:
                            st.markdown(f"   *Suggestion: {issue.suggestion}*")
                
                # Refactored code
                if result.refactored_code:
                    with st.expander("✨ Refactored Code"):
                        st.code(result.refactored_code)
                
                render_agent_steps(steps)
                
            except Exception as e:
                error_msg = str(e)
                # Handle rate limit errors
                if "429" in error_msg or "rate" in error_msg.lower() or "rate-limited" in error_msg.lower():
                    st.error("⚠️ **Rate Limit Error**: This model is temporarily rate-limited.")
                    st.info("💡 **Suggestion**: Try switching to a different model in the sidebar, or wait a few minutes and try again. Free models are more likely to be rate-limited.")
                else:
                    st.error(f"Analysis failed: {error_msg}")


# ============== Tab 3: Documentation Chat ==============
def render_documentation_tab():
    """Render the documentation chat tab."""
    st.markdown("### 📚 Documentation Chat")
    st.markdown("Ask questions about your uploaded documents.")
    
    docs = st.session_state.document_store.list_documents()
    if not docs:
        st.info("📤 Upload PDF documents in the sidebar to get started!")
        return
    
    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about your documentation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    retrieval = RetrievalTools(st.session_state.vector_store, st.session_state.document_store)
                    agent = RAGAgent(st.session_state.selected_model, retrieval)
                    response, steps = agent.answer_sync(prompt)
                    
                    st.markdown(response.answer)
                    
                    # Sources
                    if response.sources:
                        with st.expander("📎 Sources"):
                            for src in response.sources:
                                st.markdown(f"- Page {src.page_number}: *{src.content[:100]}...*")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.answer})
                    st.session_state.agent_steps = steps
                    
                except Exception as e:
                    error_msg = str(e)
                    # Handle rate limit errors
                    if "429" in error_msg or "rate" in error_msg.lower() or "rate-limited" in error_msg.lower():
                        st.error("⚠️ **Rate Limit Error**: This model is temporarily rate-limited.")
                        st.info("💡 **Suggestion**: Try switching to a different model in the sidebar, or wait a few minutes and try again. Free models are more likely to be rate-limited.")
                    else:
                        st.error(f"Error: {error_msg}")


# ============== Tab 4: Agent Logs ==============
def render_logs_tab():
    """Render the agent workflow logs tab."""
    st.markdown("### 📋 Agent Workflow Logs")
    
    if not st.session_state.agent_steps:
        st.info("Execute an agent task to see the reasoning steps here.")
        return
    
    render_agent_steps(st.session_state.agent_steps)
    
    if st.button("🗑️ Clear Logs"):
        st.session_state.agent_steps = []
        st.rerun()


# ============== Main App ==============
def main():
    """Main application entry point."""
    render_sidebar()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        st.warning("⚠️ Please set your OPENROUTER_API_KEY in .env or Streamlit secrets")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💻 Code Generation",
        "🔍 Code Analysis",
        "📚 Documentation Chat",
        "📋 Agent Logs"
    ])
    
    with tab1:
        render_code_generation_tab()
    
    with tab2:
        render_code_analysis_tab()
    
    with tab3:
        render_documentation_tab()
    
    with tab4:
        render_logs_tab()


if __name__ == "__main__":
    main()
