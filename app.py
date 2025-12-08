import streamlit as st
import os
import dotenv
from pdf_processor import process_pdf_and_upload, render_pdf_page_to_png_bytes
from chatbot_utils import process_user_query
from pinecone import Pinecone

def pinecone_index_is_empty(pinecone_api_key, pinecone_index_name):
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0) == 0

def get_index_stats(pinecone_api_key, pinecone_index_name):
    """Get comprehensive index statistics"""
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    stats = index.describe_index_stats()
    return stats

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
gemini_api_key = os.getenv("GEMINI_API_KEY")

PDF_URL = "https://raw.githubusercontent.com/Maniyuvi/CSvFile/main/om_pead-rp71-140jaa_kd79d904h01%20(1).pdf"

st.set_page_config(page_title="PDF Knowledge Assistant", layout="centered")

# ---- Enhanced CSS with better mobile support ----
st.markdown("""
    <style>
    /* Use CSS variables that adapt to Streamlit's theme */
    :root {
        --bg-primary: #f5f5f7;
        --bg-secondary: #ffffff;
        --text-primary: #1a1a1a;
        --border-color: #e0e0e0;
        --shadow: rgba(0,0,0,0.1);
    }
    
    /* Dark mode overrides */
    [data-testid="stAppViewContainer"][data-theme="dark"] {
        --bg-primary: #0e1117;
        --bg-secondary: #262730;
        --text-primary: #fafafa;
        --border-color: #4a4a4a;
        --shadow: rgba(0,0,0,0.3);
    }
    
    .fixed-title {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: var(--bg-secondary);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border-color);
        padding: 1rem;
        z-index: 999;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 2px 10px var(--shadow);
        color: var(--text-primary);
    }
    
    .main-content {
        padding-top: 80px;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        text-align: right;
        margin-bottom: 12px;
    }
    
    .bot-message {
        display: flex;
        justify-content: flex-start;
        text-align: left;
        margin-bottom: 12px;
    }
    
    .chat-bubble {
        max-width: 75%;
        padding: 14px 18px;
        border-radius: 18px;
        margin: 4px 8px;
        word-wrap: break-word;
        font-size: 1rem;
        line-height: 1.4;
        box-shadow: 0 2px 10px var(--shadow);
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #007aff, #0051d5);
        color: white;
        border-bottom-right-radius: 6px;
    }
    
    .bot-bubble {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 6px;
    }
    
    .stats-container {
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        color: var(--text-primary);
    }
    
    .upload-progress {
        background: var(--bg-secondary);
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: var(--text-primary);
    }
    
    .success-message {
        background: var(--bg-secondary);
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #4caf50;
    }
    
    .warning-message {
        background: var(--bg-secondary);
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #ff9800;
    }
    
    /* Fix Streamlit's default dark mode text colors */
    [data-theme="dark"] .stats-container h4,
    [data-theme="dark"] .stats-container p {
        color: var(--text-primary) !important;
    }

    /* Fixed FAQ Container styling */
    .faq-container {
        position: sticky;
        bottom: 0;
        background: var(--bg-secondary);
        border-top: 1px solid var(--border-color);
        padding: 12px 0;
        z-index: 100;
        box-shadow: 0 -2px 10px var(--shadow);
        margin-top: 10px;
    }

    .faq-button {
        display: inline-block;
        padding: 10px 16px;
        background: linear-gradient(135deg, #007aff, #0051d5);
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }

    .faq-button:hover {
        background: linear-gradient(135deg, #0051d5, #003d99);
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
    }

    .faq-content {
        margin-top: 12px;
        padding: 12px;
        background: var(--bg-primary);
        border-radius: 8px;
        border-left: 4px solid #007aff;
    }
    </style>
""", unsafe_allow_html=True)

@st.dialog("Source page", width="medium")
def show_source_dialog(png_bytes: bytes):
    st.image(png_bytes)

# ---- Fixed Title ----
st.markdown('<div class="fixed-title">PDF Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ---- Initialize Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_state" not in st.session_state:
    st.session_state.upload_state = "normal"
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "faq_open" not in st.session_state:
    st.session_state.faq_open = False

# ---- Sidebar with Enhanced Navigation ----
with st.sidebar:
    st.title("Navigation")
    
    # Show index statistics
    try:
        stats = get_index_stats(pinecone_api_key, pinecone_index_name)
        total_vectors = stats.get("total_vector_count", 0)
        
        if total_vectors > 0:
            st.markdown(f"""
            <div class="stats-container">
                <h4>📊 Database Status</h4>
                <p><strong>Status:</strong> Ready for queries</p>
                <p><strong>Documents:</strong> {total_vectors} chunks</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-message">
                <h4>⚠️ No Documents</h4>
                <p>Upload PDFs to get started!</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Database connection error: {e}")
    
    # Determine default page
    if total_vectors == 0:
        default_page = "Upload PDFs"
    else:
        default_page = "Chat Assistant"
    
    page = st.radio(
        "Choose Action:",
        options=["Chat Assistant", "Upload PDFs", "Database Management"],
        index=["Chat Assistant", "Upload PDFs", "Database Management"].index(default_page)
    )

# ---- Upload PDFs Page ----
if page == "Upload PDFs":
    st.header("📤 Upload PDF Documents")
    st.markdown("Upload PDF files to build your knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )
    
    if uploaded_files:
        
        if st.session_state.upload_state == "normal":
            if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
                st.session_state.upload_state = "uploading"
                st.rerun()
        elif st.session_state.upload_state == "uploading":
            st.button("⏳ Processing...", disabled=True, use_container_width=True)
        elif st.session_state.upload_state == "completed":
            if st.button("✅ Processing Completed", disabled=False, use_container_width=True):
                st.session_state.upload_state = "normal"
                st.rerun()
        elif st.session_state.upload_state == "partial":
            if st.button("⚠️ Partially Completed", disabled=False, use_container_width=True):
                st.session_state.upload_state = "normal"
                st.rerun()
        elif st.session_state.upload_state == "failed":
            if st.button("❌ Processing Failed", disabled=False, use_container_width=True):
                st.session_state.upload_state = "normal"
                st.rerun()
        
        if st.session_state.upload_state == "uploading":
            status_container = st.container()
            
            success_count = 0
            total_files = len(uploaded_files)
            processing_errors = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                with status_container:
                    st.markdown(f"""
                    <div class="upload-progress">
                        <strong>Processing:</strong> {uploaded_file.name} ({i+1}/{total_files})
                    </div>
                    """, unsafe_allow_html=True)
                # Ensure persistent PDF storage directory exists
                pdf_dir = os.getenv("PDF_DIR", "pdfs")
                os.makedirs(pdf_dir, exist_ok=True)

                # Save uploaded file using the original filename (spaces replaced)
                safe_name = uploaded_file.name.replace(' ', '_')
                saved_path = os.path.join(pdf_dir, safe_name)

                try:
                    # Save canonical file for future rendering and reference
                    with open(saved_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Use processing pipeline on the saved canonical file
                    result = process_pdf_and_upload(
                        saved_path,
                        gemini_api_key,
                        openai_api_key,
                        pinecone_api_key,
                        pinecone_index_name
                    )
                    
                    if result:
                        success_count += 1
                        st.session_state.processed_files.append({
                            'name': uploaded_file.name,
                            'status': 'success'
                        })
                    else:
                        processing_errors.append(f"{uploaded_file.name}: Processing failed")
                        st.session_state.processed_files.append({
                            'name': uploaded_file.name,
                            'status': 'error',
                            'error': 'Processing failed'
                        })
                        
                except Exception as e:
                    error_msg = f"{uploaded_file.name}: {str(e)}"
                    processing_errors.append(error_msg)
                    st.session_state.processed_files.append({
                        'name': uploaded_file.name,
                        'status': 'error',
                        'error': str(e)
                    })
                finally:
                    # Clean up temporary file
                    if os.path.exists(saved_path):
                        try:
                            os.remove(saved_path)
                        except Exception as e:
                            print(f"⚠️ Could not delete {saved_path}: {e}") 
            
            # Update upload state based on results
            if success_count == total_files:
                st.session_state.upload_state = "completed"
            elif success_count > 0:
                st.session_state.upload_state = "partial"
            else:
                st.session_state.upload_state = "failed"
            
            # Better success/error reporting
            if success_count == total_files:
                st.markdown(f"""
                <div class="success-message">
                    <h4>✅ All PDFs processed successfully!</h4>
                    <p>Processed {success_count}/{total_files} documents</p>
                    <p>You can now ask questions about your documents.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Processed {success_count}/{total_files} documents successfully.")
                if processing_errors:
                    st.error("Errors encountered:")
                    for error in processing_errors:
                        st.error(f"• {error}")
            
            # Auto-redirect to chat if successful
            if success_count > 0:
                st.success("🎉 Ready to chat! Click 'Chat Assistant' to start asking questions.")
            
            # Trigger rerun to update button state
            st.rerun()

# ---- Chat Assistant Page ----
elif page == "Chat Assistant":
    if total_vectors == 0:
        st.warning("⚠️ No documents found. Please upload some PDF documents first.")
        if st.button("📤 Upload PDFs", type="primary"):
            st.session_state.page = "Upload PDFs"
            st.rerun()
    else:
        # Chat interface
        chat_container = st.container()

        with chat_container:
            for i, msg in enumerate(st.session_state.chat_history):
                css_class = "user-message" if msg["role"] == "user" else "bot-message"
                bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
                
                bubble_html = f"""
                    <div class="{css_class}">
                        <div class="chat-bubble {bubble_class}">
                            {msg["content"]}
                        </div>
                    </div>
                """
                st.markdown(bubble_html, unsafe_allow_html=True)
                
                if msg["role"] == "assistant" and msg.get("groundings"):
                    grounding = msg["groundings"][0]
                    src = grounding.get("source")
                    page_no = grounding.get("page_number")
                    
                    if src and page_no:
                        api_base = os.getenv("IMAGE_API_BASE", "http://localhost:8000")
                        img_url = f"{api_base}/generate_image?page={int(page_no)}&zoom=2"
                        
                        if st.button("📄 View Source", key=f"view_source_{i}"):
                            png_bytes = render_pdf_page_to_png_bytes(PDF_URL, page_number=int(page_no), zoom=2.0)
                            show_source_dialog(png_bytes)

        # Chat input - FULL WIDTH
        user_input = st.chat_input("Ask about your documents...")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

            try:
                # Process query
                with st.spinner("🔍 Searching your documents..."):
                    bot_reply, source = process_user_query(user_input.strip(), st.session_state.chat_history[:-1])
                # Prepare grounding metadata list from matches
                groundings = []
                if source.get('source') and source.get('page'):
                    groundings.append({
                        'source': source.get('source'),
                        'page_number': source.get('page')
                    })

                # Add bot response to history with grounding metadata
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "groundings": groundings})

            except Exception as ex:
                error_msg = f"Sorry, I encountered an error: {str(ex)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.error(f"Error processing query: {ex}")

            st.rerun()

        # Frequently Asked Questions / Suggestions (expander - initially closed)
        with st.expander("💡 Frequently Asked Questions / Suggestions", expanded=st.session_state.faq_open):
            suggested_questions = [
                "What are the key safety procedures described in the documents?",
                "Summarize maintenance schedule guidelines.",
                "What are the contact details for emergency?",
                "Available temperature ranges?"
            ]

            cols = st.columns(3)
            for i, q in enumerate(suggested_questions):
                col = cols[i % len(cols)]
                if col.button(q, key=f"suggestion_{i}"):
                    # When a suggestion is clicked, add as user input and process it
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    try:
                        with st.spinner("🔍 Searching your documents..."):
                            bot_reply, source = process_user_query(q, st.session_state.chat_history[:-1])

                            groundings = []
                            if source.get('source') and source.get('page'):
                                groundings.append({
                                    'source': source.get('source'),
                                    'page_number': source.get('page')
                                })

                            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply, "groundings": groundings})
                    except Exception as ex:
                        error_msg = f"Sorry, I encountered an error: {str(ex)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        st.error(f"Error processing suggestion: {ex}")

                    # Auto-close the FAQ and rerun to show updated chat
                    st.session_state.faq_open = False
                    st.rerun()


# ---- Database Management Page ----
elif page == "Database Management":
    st.header("🛠️ Database Management")
    
    # Show detailed statistics
    try:
        stats = get_index_stats(pinecone_api_key, pinecone_index_name)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats.get("total_vector_count", 0))
        with col2:
            st.metric("Index Dimension", stats.get("dimension", "N/A"))
        
        # Show recently processed files
        if st.session_state.processed_files:
            st.subheader("📋 Recently Processed Files")
            
            for file_info in st.session_state.processed_files[-20:]:  # Show last 20
                status_icon = "✅" if file_info['status'] == 'success' else "❌"
                error_text = f" ({file_info.get('error', 'Unknown error')})" if file_info['status'] == 'error' else ""
                st.markdown(f"{status_icon} {file_info['name']}{error_text}")
        
    except Exception as e:
        st.error(f"Error retrieving database statistics: {e}")
    
    # Database operations
    st.subheader("🗃️ Database Operations")
    
    # Clear chat history
    if st.button("🧹 Clear Chat History", help="Clear conversation history (keeps documents)"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    # Reset entire database
    st.subheader("⚠️ Reset Database")
    st.markdown("**Warning:** This will delete all uploaded documents and cannot be undone.")
    
    reset_confirm = st.checkbox("I understand this will delete all documents")
    
    if reset_confirm:
        if st.button("🗑️ Reset Entire Database", type="secondary"):
            with st.spinner("Resetting database..."):
                try:
                    pc = Pinecone(api_key=pinecone_api_key)
                    index = pc.Index(pinecone_index_name)
                    index.delete(delete_all=True)
                    
                    # Clear session state
                    st.session_state.chat_history = []
                    st.session_state.processed_files = []
                    st.session_state.upload_state = "normal"
                    
                    st.success("✅ Database reset successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error resetting database: {e}")

st.markdown('</div>', unsafe_allow_html=True)
