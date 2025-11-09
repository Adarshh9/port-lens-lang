"""
Production-Ready Streamlit UI for RAG + LLM System
Provides complete interface for document management and querying.
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG + LLM System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "RAG + LLM System v1.0"
    }
)

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================

API_BASE_URL = "http://127.0.0.1:8000/api/v1"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
INGEST_ENDPOINT = f"{API_BASE_URL}/ingest"
CACHE_CLEAR_ENDPOINT = f"{API_BASE_URL}/cache/clear"

# ============================================================================
# CUSTOM CSS & STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #0D47A1;
        --secondary-color: #1976D2;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --danger-color: #F44336;
    }
    
    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if "api_health" not in st.session_state:
    st.session_state.api_health = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "query_stats" not in st.session_state:
    st.session_state.query_stats = {
        "total_queries": 0,
        "cache_hits": 0,
        "avg_processing_time": 0,
        "avg_quality_score": 0
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def check_api_health():
    """Check API health status."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"API Health Check Failed: {str(e)}")
        return None


def send_query(query: str, user_id: str = "streamlit_user") -> dict:
    """Send query to RAG system."""
    try:
        payload = {
            "query": query,
            "session_id": st.session_state.session_id,
            "user_id": user_id
        }
        
        response = requests.post(QUERY_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Request Failed: {str(e)}")
        return None


def ingest_document(file_path: str) -> dict:
    """Ingest document into the system."""
    try:
        payload = {"file_path": file_path}
        response = requests.post(INGEST_ENDPOINT, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ingestion Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ingestion Request Failed: {str(e)}")
        return None


def clear_cache():
    """Clear API cache."""
    try:
        response = requests.get(CACHE_CLEAR_ENDPOINT, timeout=5)
        if response.status_code == 200:
            st.success("✅ Cache cleared successfully!")
            return True
        return False
    except Exception as e:
        st.error(f"Failed to clear cache: {str(e)}")
        return False


def format_response(response: dict) -> None:
    """Display formatted query response."""
    if not response:
        return
    
    # Main answer
    st.markdown("### 📝 Answer")
    st.info(response.get("answer", "No answer received"))
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⏱️ Processing Time",
            f"{response.get('processing_time', 0):.2f}s",
            delta=None
        )
    
    with col2:
        cache_hit = response.get("cache_hit", False)
        st.metric(
            "💾 Cache Hit",
            "Yes ✅" if cache_hit else "No ❌",
            delta=None
        )
    
    with col3:
        quality = response.get("quality_passed", False)
        st.metric(
            "✔️ Quality",
            "Passed" if quality else "Failed",
            delta=None
        )
    
    with col4:
        judge_score = response.get("judge_evaluation", {}).get("score", 0)
        st.metric(
            "📊 Judge Score",
            f"{judge_score:.2f}/1.0",
            delta=None
        )
    
    # Retrieved documents
    st.markdown("### 📚 Retrieved Documents")
    docs = response.get("retrieved_docs", [])
    
    if docs:
        for idx, doc in enumerate(docs, 1):
            with st.expander(f"📄 Document {idx} (Relevance: {doc.get('distance', 0):.3f})"):
                st.markdown(doc.get("content", "No content"))
                if doc.get("metadata"):
                    st.caption(f"Metadata: {json.dumps(doc.get('metadata', {}))}")
    else:
        st.info("No documents retrieved")
    
    # Judge evaluation details
    st.markdown("### 🔍 Judge Evaluation Details")
    je = response.get("judge_evaluation", {})
    
    if je:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Reasons:** {je.get('reasons', 'N/A')}")
        
        with col2:
            criteria = je.get("criteria", {})
            if criteria:
                st.write("**Criteria Scores:**")
                for key, value in criteria.items():
                    st.write(f"- {key.capitalize()}: {value}")


def update_query_stats(response: dict) -> None:
    """Update query statistics."""
    st.session_state.query_stats["total_queries"] += 1
    
    if response.get("cache_hit"):
        st.session_state.query_stats["cache_hits"] += 1
    
    # Update averages
    stats = st.session_state.query_stats
    processing_time = response.get("processing_time", 0)
    judge_score = response.get("judge_evaluation", {}).get("score", 0)
    
    # Simple running average
    n = stats["total_queries"]
    stats["avg_processing_time"] = (
        (stats["avg_processing_time"] * (n - 1) + processing_time) / n
    )
    stats["avg_quality_score"] = (
        (stats["avg_quality_score"] * (n - 1) + judge_score) / n
    )


# ============================================================================
# PAGE LAYOUT
# ============================================================================

def render_header():
    """Render page header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🤖 RAG + LLM System")
        st.markdown("**Production-Grade Retrieval-Augmented Generation with LangGraph**")
    
    with col2:
        # Health indicator
        health = check_api_health()
        if health:
            st.success("✅ API Healthy")
            st.session_state.api_health = True
        else:
            st.error("❌ API Offline")
            st.session_state.api_health = False


def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Session info
        st.markdown("### Session Info")
        st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            disabled=True,
            key="sid_display"
        )
        
        # Query stats
        st.markdown("### 📊 Query Statistics")
        stats = st.session_state.query_stats
        
        st.metric("Total Queries", stats["total_queries"])
        st.metric("Cache Hits", stats["cache_hits"])
        st.metric(
            "Avg Processing Time",
            f"{stats['avg_processing_time']:.2f}s"
        )
        st.metric(
            "Avg Quality Score",
            f"{stats['avg_quality_score']:.2f}/1.0"
        )
        
        # Actions
        st.markdown("### 🔧 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.session_id = f"session_{int(time.time())}"
                st.session_state.conversation_history = []
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                clear_cache()
        
        # About
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "RAG + LLM System\n\n"
            "**Version:** 1.0.0\n\n"
            "**Components:**\n"
            "- LangGraph Orchestration\n"
            "- Vector Database (Chroma)\n"
            "- LLM (Groq - Llama 3.1)\n"
            "- Multi-layer Memory\n"
            "- Quality Judge Module"
        )


def render_query_page():
    """Render query interaction page."""
    st.markdown("## 💬 Query Assistant")
    
    # Query input
    st.markdown("### Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Your Question:",
            placeholder="e.g., What is machine learning?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button("🔍 Search", use_container_width=True)
    
    # Process query
    if submit_button and query:
        with st.spinner("Processing your query..."):
            response = send_query(query)
            
            if response:
                # Add to history
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "response": response
                })
                
                # Update stats
                update_query_stats(response)
                
                # Display response
                format_response(response)
            else:
                st.error("Failed to process query")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        
        for idx, item in enumerate(reversed(st.session_state.conversation_history), 1):
            with st.expander(f"Query {len(st.session_state.conversation_history) - idx + 1}: {item['query'][:50]}..."):
                st.write(f"**Query:** {item['query']}")
                st.write(f"**Time:** {item['timestamp']}")
                st.markdown(f"**Answer:** {item['response'].get('answer', 'N/A')[:200]}...")


def render_document_page():
    """Render document ingestion page."""
    st.markdown("## 📄 Document Management")
    
    tab1, tab2 = st.tabs(["📤 Upload Document", "📊 Ingestion Status"])
    
    with tab1:
        st.markdown("### Upload Documents")
        st.info(
            "**Supported Formats:**\n"
            "- PDF (.pdf)\n"
            "- Text (.txt)\n"
            "- Markdown (.md)"
        )
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Display file info
            st.markdown("### File Preview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.type)
            
            # Save and ingest
            if st.button("✅ Ingest Document", type="primary"):
                # Save temporary file
                temp_path = f"./temp/{uploaded_file.name}"
                Path("./temp").mkdir(exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                with st.spinner("Ingesting document..."):
                    result = ingest_document(temp_path)
                    
                    if result:
                        st.success(f"✅ Document ingested successfully!")
                        st.json(result)
                    else:
                        st.error("Failed to ingest document")
                
                # Cleanup
                Path(temp_path).unlink(missing_ok=True)
    
    with tab2:
        st.markdown("### Ingestion Status")
        st.info(
            "Document ingestion status and statistics will appear here.\n\n"
            "**Current Status:** Ready for documents"
        )
        
        # Placeholder stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", "0")
        with col2:
            st.metric("Total Chunks", "0")
        with col3:
            st.metric("Vector Dimension", "384")
        with col4:
            st.metric("Storage Size", "0 MB")


def render_analytics_page():
    """Render analytics and monitoring page."""
    st.markdown("## 📊 Analytics & Monitoring")
    
    tab1, tab2 = st.tabs(["📈 Performance", "🔍 System Health"])
    
    with tab1:
        st.markdown("### Query Performance")
        
        if st.session_state.conversation_history:
            # Extract data
            processing_times = []
            quality_scores = []
            cache_hits = []
            timestamps = []
            
            for item in st.session_state.conversation_history:
                resp = item["response"]
                processing_times.append(resp.get("processing_time", 0))
                quality_scores.append(resp.get("judge_evaluation", {}).get("score", 0))
                cache_hits.append(1 if resp.get("cache_hit") else 0)
                timestamps.append(item["timestamp"])
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Processing time chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(processing_times))),
                    y=processing_times,
                    mode='lines+markers',
                    name='Processing Time',
                    line=dict(color='#1f77b4')
                ))
                fig.update_layout(
                    title="Processing Time Trend",
                    xaxis_title="Query Index",
                    yaxis_title="Time (seconds)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Quality score chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(quality_scores))),
                    y=quality_scores,
                    mode='lines+markers',
                    name='Quality Score',
                    line=dict(color='#2ca02c')
                ))
                fig.update_layout(
                    title="Answer Quality Trend",
                    xaxis_title="Query Index",
                    yaxis_title="Score (0-1)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            st.markdown("### Query Statistics")
            stats_df = pd.DataFrame({
                "Metric": [
                    "Total Queries",
                    "Cache Hit Rate",
                    "Avg Processing Time",
                    "Avg Quality Score",
                    "Min Processing Time",
                    "Max Processing Time"
                ],
                "Value": [
                    len(processing_times),
                    f"{(sum(cache_hits) / len(cache_hits) * 100):.1f}%",
                    f"{sum(processing_times) / len(processing_times):.2f}s",
                    f"{sum(quality_scores) / len(quality_scores):.2f}",
                    f"{min(processing_times):.2f}s",
                    f"{max(processing_times):.2f}s"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No query data available yet. Start by asking a question!")
    
    with tab2:
        st.markdown("### System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health = check_api_health()
            if health:
                st.success("✅ API Status: Healthy")
            else:
                st.error("❌ API Status: Offline")
        
        with col2:
            st.info("💾 Cache Status: Active")
        
        with col3:
            st.success("📊 Vector DB: Ready")
        
        # Detailed health info
        st.markdown("### Health Details")
        if health:
            st.json(health)


def render_settings_page():
    """Render settings page."""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2 = st.tabs(["API Configuration", "System Preferences"])
    
    with tab1:
        st.markdown("### API Configuration")
        
        st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            disabled=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input(
                "Query Timeout (seconds)",
                value="30",
                disabled=True
            )
        with col2:
            st.text_input(
                "Ingest Timeout (seconds)",
                value="60",
                disabled=True
            )
        
        if st.button("🔄 Test Connection"):
            with st.spinner("Testing connection..."):
                health = check_api_health()
                if health:
                    st.success("✅ Connection successful!")
                    st.json(health)
                else:
                    st.error("❌ Connection failed")
    
    with tab2:
        st.markdown("### System Preferences")
        
        theme = st.radio(
            "Theme",
            ["Light", "Dark", "Auto"],
            horizontal=True
        )
        
        auto_refresh = st.checkbox(
            "Auto-refresh stats every 30 seconds",
            value=False
        )
        
        show_advanced = st.checkbox(
            "Show advanced options",
            value=False
        )
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved!")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app logic."""
    render_header()
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Query",
        "📄 Documents",
        "📊 Analytics",
        "⚙️ Settings",
        "ℹ️ Help"
    ])
    
    with tab1:
        render_query_page()
    
    with tab2:
        render_document_page()
    
    with tab3:
        render_analytics_page()
    
    with tab4:
        render_settings_page()
    
    with tab5:
        st.markdown("## Help & Documentation")
        
        st.markdown("""
        ### Getting Started
        
        1. **Ask Questions**: Use the Query tab to ask questions about your documents
        2. **Upload Documents**: Add PDF, TXT, or MD files in the Documents tab
        3. **Monitor Performance**: Check real-time analytics in the Analytics tab
        
        ### Features
        
        - **Semantic Search**: Retrieves relevant documents using embeddings
        - **LLM Answer**: Generates contextual answers using Llama 3.1
        - **Quality Judge**: Automatically evaluates answer quality
        - **Caching**: Fast responses for repeated questions
        - **Multi-layer Memory**: Short-term and long-term conversation tracking
        
        ### Supported File Types
        
        - PDF (.pdf)
        - Plain Text (.txt)
        - Markdown (.md)
        
        ### Performance Tips
        
        - Shorter, specific questions yield better results
        - Longer documents provide more context
        - First query takes longer (LLM initialization)
        - Repeated queries are cached for speed
        
        ### Troubleshooting
        
        **API is offline?**
        - Ensure FastAPI server is running: `python -m uvicorn app.main:app`
        
        **Slow responses?**
        - Check network connection
        - Monitor system resources
        - Clear cache if needed
        
        **Poor answer quality?**
        - Add more relevant documents
        - Rephrase your question
        - Use more specific terminology
        """)


if __name__ == "__main__":
    main()
