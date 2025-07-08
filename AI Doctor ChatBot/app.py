import os
import streamlit as st
from medbot import MedBotApp

# Set page configuration
st.set_page_config(
    page_title="MedBot - Medical Knowledge Assistant",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)

def initialize_session_state():
    """Initialize session state variables."""
    if "medbot" not in st.session_state:
        st.session_state.medbot = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "index_mode" not in st.session_state:
        st.session_state.index_mode = "existing"
    if "query_type" not in st.session_state:
        st.session_state.query_type = "medical"

def display_chat_history():
    """Display chat history with modern formatting."""
    # Clear previous chat messages from the container
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🩺"):
                # Display main answer
                st.write(message["content"])

def create_sidebar():
    """Create simplified sidebar with configuration options."""
    with st.sidebar:
        st.markdown(
            """
        <div style='text-align: center; padding: 0.5rem 0;'>
            <h2 style='color: #2b5198;'>🩺 MedBot</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Index selection
        index_mode = st.radio(
            "Choose Index Mode",
            ["Use existing index", "Upload new documents"],
            index=0 if st.session_state.index_mode == "existing" else 1,
        )
        st.session_state.index_mode = (
            "existing" if index_mode == "Use existing index" else "new"
        )

        # Document Upload Section (only show if "Upload new documents" is selected)
        if st.session_state.index_mode == "new":
            uploaded_files = st.file_uploader(
                "Upload Medical Documents (PDF)", type="pdf", accept_multiple_files=True
            )
        else:
            uploaded_files = []

        # Model Settings
        st.subheader("Model Configuration")
        model_choice = st.selectbox(
            "LLM Model",
            ["mixtral-8x7b-32768", "deepseek-r1-distill-qwen-32b", "llama2-70b-4096"],
            index=0,
        )
        search_k = st.slider("Number of Document Matches", 1, 5, 3)
        query_type = st.selectbox(
            "Query Type", ["medical", "educational", "detailed", "general"], index=0
        )
        st.session_state.query_type = query_type

        # Initialize Button
        if st.session_state.index_mode == "existing":
            if st.button(
                "Initialize with Existing Index",
                type="primary",
                use_container_width=True,
            ):
                initialize_with_existing_index(model_choice, search_k)
        else:
            if st.button("Process Documents", type="primary", use_container_width=True):
                if uploaded_files:
                    process_documents(uploaded_files, model_choice, search_k)
                else:
                    st.error("Please upload at least one PDF document")

        # Clear chat button
        if st.session_state.chat_history and st.button(
            "🧹 Clear Chat", use_container_width=True
        ):
            st.session_state.chat_history = []
            st.rerun()

def initialize_with_existing_index(model_name, search_k):
    """Initialize MedBot with an existing index."""
    try:
        with st.spinner("🔍 Loading existing index..."):
            medbot = MedBotApp()
            medbot.setup(model_name, create_new_index=False)

            # Configure retriever
            if hasattr(medbot, "doc_chat") and hasattr(medbot.doc_chat, "retriever"):
                medbot.doc_chat.retriever.search_kwargs["k"] = search_k

            st.session_state.medbot = medbot
            st.session_state.docs_processed = True

            st.success("✅ Successfully loaded existing index!")
    except Exception as e:
        st.error(f"Error loading existing index: {str(e)}")

def process_documents(uploaded_files, model_name, search_k):
    """Handle document processing and MedBot initialization with new documents."""
    try:
        # Save uploaded files temporarily
        data_dir = "temp_docs"
        os.makedirs(data_dir, exist_ok=True)

        for file in uploaded_files:
            file_path = os.path.join(data_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        # Initialize MedBot
        with st.spinner("🔍 Processing medical documents..."):
            medbot = MedBotApp(data_dir=data_dir)
            medbot.setup(model_name, create_new_index=True)
            medbot.doc_chat.retriever.search_kwargs["k"] = search_k

            st.session_state.medbot = medbot
            st.session_state.docs_processed = True
            st.session_state.uploaded_files = [f.name for f in uploaded_files]

            st.success(f"✅ Processed {len(uploaded_files)} documents!")
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                os.remove(os.path.join(data_dir, f))
            os.rmdir(data_dir)

def main_app():
    """Main application interface."""
    initialize_session_state()

    # Create sidebar
    create_sidebar()

    # Main content area
    st.markdown(
        """
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #2b5198;'>🩺 MedBot</h1>
        <p style='color: #666;'>Medical Knowledge Retrieval & Consultation System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.docs_processed:
        if st.session_state.index_mode == "existing":
            st.info("👋 Please click 'Initialize with Existing Index' to begin")
        else:
            st.info(
                "👋 Please upload medical documents and click 'Process Documents' to begin"
            )
        return

    # Display processed documents if any were uploaded
    if st.session_state.index_mode == "new" and st.session_state.uploaded_files:
        with st.expander("📚 Loaded Documents", expanded=False):
            st.write(f"Processed {len(st.session_state.uploaded_files)} documents:")
            for doc in st.session_state.uploaded_files:
                st.markdown(f"- {doc}")
    elif st.session_state.index_mode == "existing":
        st.success("Using existing document index")

    # Chat container - define it once and reuse
    chat_container = st.container()

    # Display chat history in the container
    with chat_container:
        display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Get MedBot response
        with st.spinner("🔬 Analyzing your question..."):
            try:
                if st.session_state.medbot is None:
                    st.error(
                        "MedBot is not initialized. Please initialize with an index first."
                    )
                    return

                response = st.session_state.medbot.query(
                    question=prompt, query_type=st.session_state.query_type
                )

                # Format response - removed sources from chat history
                assistant_msg = {
                    "role": "assistant",
                    "content": response["answer"],
                }
                st.session_state.chat_history.append(assistant_msg)

                # Use st.rerun() to properly refresh the page with the new chat history
                st.rerun()

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.rerun()

if __name__ == "__main__":
    main_app()