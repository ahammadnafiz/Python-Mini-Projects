# app.py

import streamlit as st
import os
from typing import Dict, Any
from dotenv import load_dotenv
from reporag.reporag.main import main

# Set page configuration
st.set_page_config(
    page_title="RepoRAG - Modern Repository Analysis",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern container styling */
    .stApp {
        background-color:#171718;
    }

    /* Header styling */
    .main-header {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #eff7f6;
        padding: 1.5rem 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 2rem;
    }

    /* Chat container styling */
    .chat-container {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #232325;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }

    /* Status indicator styling */
    .status-indicator {
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: 500;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .user-message {
        background-color: #f0f7ff;
    }

    .assistant-message {
        background-color: #252422;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_instance' not in st.session_state:
        st.session_state.rag_instance = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False
    if 'repository_processed' not in st.session_state:
        st.session_state.repository_processed = False
    if 'output_file' not in st.session_state:
        st.session_state.output_file = "prompt.txt"

def validate_repo_url(url: str) -> bool:
    """Validate the repository URL format."""
    parts = url.split('/')
    return len(parts) == 2 and all(part.strip() for part in parts)

def display_chat_history():
    """Display chat history with modern formatting."""
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            with st.chat_message("user", avatar="👤"):
                st.write(message)
        else:  # Assistant message
            with st.chat_message("assistant", avatar="🤖"):
                if isinstance(message, dict):
                    # Main answer with modern formatting
                    # st.markdown(f"""
                    # <div class='chat-message assistant-message'>
                    #     {message["answer"]}
                    # </div>
                    # """, unsafe_allow_html=True)
                    st.write(message["answer"])

                    # Sources in a modern expander
                    if message.get("sources"):
                        with st.expander("📚 View Sources", expanded=False):
                            cols = st.columns(2)
                            for idx, source in enumerate(message["sources"]):
                                col = cols[idx % 2]
                                with col:
                                    st.markdown(f"""
                                    <div style='background-color: #212529; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                                        <p><strong>Source {idx + 1}</strong></p>
                                        <p><code>{source['metadata'].get('file_path', 'unknown')}</code></p>
                                        <p>Language: {source['metadata'].get('language', 'unknown')}</p>
                                        <p style='font-size: 0.9em; color: #666;'>{source['content']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.markdown(message)

def create_sidebar():
    """Create a modern sidebar with configuration options."""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #f4f1de; font-size: 1.5rem;'>⚙️ Configuration</h2>
        </div>
        """, unsafe_allow_html=True)

        # Create tabs for different settings
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Basic", "🔧 Advanced", "🧹 Cleanup", "📊 Status"])

        with tab1:
            # Repository URL input
            repo_url = st.text_input(
                "Repository URL",
                placeholder="owner/repo",
                help="Enter in format: owner/repo"
            )

            # Token inputs with modern styling
            github_token = st.text_input(
                "GitHub Token",
                type="password",
                value=os.getenv('GITHUB_ACCESS_TOKEN', ''),
                help="Your GitHub personal access token"
            )

            google_token = st.text_input(
                "Google API Key",
                type="password",
                value=os.getenv('GOOGLE_API_KEY', ''),
                help="Your Google AI API key for Gemini"
            )

            # Output file input with file extension validation
            output_file = st.text_input(
                "Output File Name",
                value=st.session_state.output_file,
                help="Enter the name for the output file (e.g., output.txt)"
            )

            if output_file and not output_file.endswith('.txt'):
                output_file += '.txt'
            st.session_state.output_file = output_file

        with tab2:
            # System statistics if initialized
            if st.session_state.is_initialized:
                st.markdown("### 📊 System Stats")
                stats = st.session_state.rag_instance.get_stats()

                # Display stats in a modern grid
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", stats['total_documents'])
                    st.metric("Chat Messages", stats['chat_history_length'])

                with col2:
                    st.metric("Total Tokens", stats['total_tokens'])
                    st.metric("Cache Size", f"{stats['cache_size'] / 1024:.2f} KB")

                # Content types in a collapsible section
                with st.expander("📑 Content Types"):
                    for content_type, count in stats['content_types'].items():
                        st.markdown(f"- **{content_type}**: {count}")

        with tab3:
            # Cleanup buttons
            if st.session_state.is_initialized:
                st.markdown("### 🧹 Clear Individual Components")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💬 Clear Chat History", use_container_width=True):
                        st.session_state.chat_history = []
                        if st.session_state.rag_instance:
                            st.session_state.rag_instance.clear_memory()
                        st.success("💫 Chat history cleared!")

                    if st.button("🗄️ Clear Vector Store", use_container_width=True):
                        if st.session_state.rag_instance:
                            st.session_state.rag_instance.clear_vector_store()
                            st.success("💫 Vector store cleared!")

                with col2:
                    if st.button("💾 Clear Cache", use_container_width=True):
                        if st.session_state.rag_instance:
                            st.session_state.rag_instance.clear_cache()
                            st.success("💫 Cache cleared!")

                    if st.button("🗂️ Clear Vector DB", use_container_width=True):
                        if st.session_state.rag_instance:
                            st.session_state.rag_instance.clear_vector_database()
                            st.success("💫 Vector database cleared - ready for new repo!")

                st.markdown("### 🔄 Repository Management")
                
                if st.button("🚀 Switch Repository Mode", use_container_width=True, type="primary"):
                    if st.session_state.rag_instance:
                        st.session_state.rag_instance.clear_vector_database()
                        st.session_state.repository_processed = False
                        st.success("✨ Ready to process a new repository!")
                        
                if st.button("💥 Complete System Reset", use_container_width=True):
                    if st.session_state.rag_instance:
                        st.session_state.rag_instance.reset_system()
                        st.session_state.chat_history = []
                        st.session_state.repository_processed = False
                        st.success("💫 Complete system reset performed!")

        with tab4:
            # System status and repository information
            if st.session_state.is_initialized and st.session_state.rag_instance:
                st.markdown("### 🎯 System Status")
                
                # Get system status
                try:
                    status = st.session_state.rag_instance.get_system_status()
                    repo_info = st.session_state.rag_instance.get_repository_info()
                    
                    # Status indicators with modern styling
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Ready status
                        ready_icon = "✅" if status["ready_for_queries"] else "❌"
                        st.markdown(f"**Query Ready:** {ready_icon}")
                        
                        # Vector store status
                        vs_icon = "✅" if status["vector_store_loaded"] else "❌"
                        st.markdown(f"**Vector Store:** {vs_icon}")
                    
                    with col2:
                        # QA Chain status
                        qa_icon = "✅" if status["qa_chain_initialized"] else "❌"
                        st.markdown(f"**QA Chain:** {qa_icon}")
                        
                        # Document count
                        doc_count = repo_info.get("document_count", 0)
                        st.markdown(f"**Documents:** {doc_count}")
                    
                    # Detailed repository information
                    with st.expander("📂 Repository Details", expanded=False):
                        st.json(repo_info)
                    
                    # System statistics
                    with st.expander("📈 Detailed Statistics", expanded=False):
                        stats = status["system_stats"]
                        
                        # Create metrics grid
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Total Documents", stats.get('total_documents', 0))
                            st.metric("Chat History", stats.get('chat_history_length', 0))
                        
                        with metric_col2:
                            st.metric("Total Tokens", stats.get('total_tokens', 0))
                            cache_size_kb = stats.get('cache_size', 0) / 1024
                            st.metric("Cache Size", f"{cache_size_kb:.2f} KB")
                        
                        with metric_col3:
                            content_types = len(stats.get('content_types', {}))
                            st.metric("Content Types", content_types)
                            languages = len(stats.get('languages', {}))
                            st.metric("Languages", languages)
                        
                        # Content breakdown
                        if stats.get('content_types'):
                            st.markdown("**Content Types:**")
                            for content_type, count in stats['content_types'].items():
                                st.markdown(f"- {content_type}: {count}")
                        
                        if stats.get('languages'):
                            st.markdown("**Languages Detected:**")
                            for language, count in stats['languages'].items():
                                st.markdown(f"- {language}: {count}")
                
                except Exception as e:
                    st.error(f"Error getting system status: {e}")
            else:
                st.info("🔧 Initialize the system to view status information")

        # Initialize/Reset button with modern styling
        if st.session_state.is_initialized and st.session_state.repository_processed:
            # If system is already initialized, offer options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("� Switch Repository", type="primary", use_container_width=True):
                    if not all([repo_url, github_token, google_token, output_file]):
                        st.error("Please fill in all required fields")
                    elif not validate_repo_url(repo_url):
                        st.error("Invalid repository format. Please use 'owner/repo' format.")
                    else:
                        with st.spinner("🔄 Switching to new repository..."):
                            try:
                                # Clear current repository data
                                st.session_state.rag_instance.clear_vector_database()
                                st.session_state.chat_history = []
                                
                                # Process new repository
                                result = main(
                                    repo_url=repo_url,
                                    access_token=github_token,
                                    google_api_key=google_token,  # Using Google API key
                                    output_file=output_file,
                                    rag_mode=True
                                )

                                if isinstance(result, str) and result.startswith("Error"):
                                    st.error(result)
                                else:
                                    st.session_state.rag_instance = result
                                    st.success("✨ Successfully switched to new repository!")
                                    st.session_state.chat_history = []
                            except Exception as e:
                                st.error(f"❌ Error switching repository: {str(e)}")
            
            with col2:
                if st.button("💥 Complete Reset", use_container_width=True):
                    try:
                        if st.session_state.rag_instance:
                            st.session_state.rag_instance.reset_system()
                        st.session_state.rag_instance = None
                        st.session_state.is_initialized = False
                        st.session_state.repository_processed = False
                        st.session_state.chat_history = []
                        st.success("💫 System completely reset!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error resetting system: {str(e)}")
        else:
            # Initial setup button
            if st.button("🚀 Initialize System", type="primary", use_container_width=True):
                if not all([repo_url, github_token, google_token, output_file]):
                    st.error("Please fill in all required fields")
                elif not validate_repo_url(repo_url):
                    st.error("Invalid repository format. Please use 'owner/repo' format.")
                else:
                    with st.spinner("🔄 Initializing system..."):
                        try:
                            result = main(
                                repo_url=repo_url,
                                access_token=github_token,
                                google_api_key=google_token,  # Using Google API key
                                output_file=output_file,
                                rag_mode=True
                            )

                            if isinstance(result, str) and result.startswith("Error"):
                                st.error(result)
                            else:
                                st.session_state.rag_instance = result
                                st.session_state.is_initialized = True
                                st.session_state.repository_processed = True
                                st.success("✨ System initialized successfully!")
                                st.session_state.chat_history = []
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

def main_app():
    """Main Streamlit application with modern layout."""
    initialize_session_state()
    load_dotenv(override=True)

    # Create sidebar
    create_sidebar()

    st.markdown("""
    <style>
        .main-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin: 20px 0;
        }
        .main-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            color: #eff7f6;
        }
        .main-header p {
            font-size: 1.2em;
            color: #ccc5b9;
        }
    </style>
    <div class='main-header'>
        <h1>
            <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.54 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.56 7.56 0 012.01-.27c.68 0 1.37.09 2.01.27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38C13.71 14.54 16 11.54 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            RepoRAG
        </h1>
        <p>Your intelligent repository analysis companion</p>
    </div>
""", unsafe_allow_html=True)

    # Status indicator
    if not st.session_state.is_initialized:
        st.warning("🚨 Please initialize the system using the sidebar configuration.")
    else:
        # Show current repository status
        if st.session_state.repository_processed and st.session_state.rag_instance:
            try:
                status = st.session_state.rag_instance.get_system_status()
                repo_info = st.session_state.rag_instance.get_repository_info()
                
                # Current repository indicator
                status_color = "#28a745" if status["ready_for_queries"] else "#dc3545"
                doc_count = repo_info.get("document_count", 0)
                
                st.markdown(f"""
                <div style='background-color: {status_color}20; border-left: 4px solid {status_color}; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                    <strong>📂 Repository Status:</strong> 
                    {"✅ Ready for queries" if status["ready_for_queries"] else "❌ Not ready"} 
                    | 📄 Documents: {doc_count} 
                    | 💬 Chat History: {repo_info.get("chat_history_length", 0)} messages
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"⚠️ System initialized but status check failed: {e}")
        else:
            st.info("ℹ️ System initialized. Please process a repository to start chatting.")

        # Modern getting started guide
        st.markdown("""
        <div style='background-color: #252422; padding: 2rem; border-radius: 10px; margin: 2rem 0;'>
            <h3>🚀 Getting Started</h3>
            <ol style='margin-top: 1rem;'>
                <li>Enter your GitHub repository URL (format: owner/repo)</li>
                <li>Provide your GitHub access token</li>
                <li>Add your Google AI API key (for Gemini)</li>
                <li>Set your desired output file name</li>
                <li>Click 'Initialize System' to start</li>
            </ol>
            
            <h4 style='margin-top: 2rem;'>🔄 Repository Switching</h4>
            <p>Once initialized, you can:</p>
            <ul>
                <li><strong>Switch Repository:</strong> Process a new repository while keeping system settings</li>
                <li><strong>Clear Vector DB:</strong> Remove current repository data to prepare for a new one</li>
                <li><strong>Complete Reset:</strong> Start fresh with a clean system</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Chat interface with modern styling - only show if system is ready
    if st.session_state.is_initialized and st.session_state.repository_processed and st.session_state.rag_instance:
        st.markdown("""
        <div style='margin: 2rem 0;'>
            <h3>💬 Chat Interface</h3>
        </div>
        """, unsafe_allow_html=True)

        # Display chat history
        with st.container():
            display_chat_history()

        # Chat input with modern styling
        if prompt := st.chat_input("💭 Ask me about the repository..."):
            if st.session_state.rag_instance:
                with st.spinner("🤔 Processing..."):
                    try:
                        st.session_state.chat_history.append(prompt)
                        response = st.session_state.rag_instance.query(prompt)
                        st.session_state.chat_history.append(response)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("⚠️ System not initialized. Please initialize first.")

        # Quick actions for repository management
        with st.expander("🔧 Quick Repository Actions", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Clear for New Repo", use_container_width=True):
                    if st.session_state.rag_instance:
                        st.session_state.rag_instance.clear_vector_database()
                        st.session_state.chat_history = []
                        st.success("✨ Ready for new repository!")
                        st.rerun()
            
            with col2:
                if st.button("💾 Save Vector Store", use_container_width=True):
                    if st.session_state.rag_instance:
                        try:
                            st.session_state.rag_instance.save_vector_store()
                            st.success("💾 Vector store saved!")
                        except Exception as e:
                            st.error(f"Error saving: {e}")
            
            with col3:
                if st.button("📊 Refresh Status", use_container_width=True):
                    st.rerun()

if __name__ == "__main__":
    main_app()