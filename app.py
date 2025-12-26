import streamlit as st
import os
import dotenv
import pandas as pd
from pdf_processor import process_pdf_and_upload, render_pdf_page_to_png_bytes
from chatbot_utils import process_user_query, transcribe_audio, generate_audio_response
from pinecone import Pinecone
import base64
import io
from urllib.parse import unquote

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

URL_LIST = [
                "https://raw.githubusercontent.com/Maniyuvi/CSvFile/main/om_pead-rp71-140jaa_kd79d904h01%20(1).pdf", 
                "https://raw.githubusercontent.com/Maniyuvi/CSvFile/main/nmc110.pdf"
            ]

def normalize(name: str):
    name = unquote(name)
    name = name.replace(" ", "_")
    return name.lower()

if "header_name" not in st.session_state:
    st.session_state.header_name = "Etihad Rail"
if "gemini_upload" not in st.session_state:
    st.session_state.gemini_upload = False
if "category" not in st.session_state:
    st.session_state.category = "HVAC"

if "category" in st.session_state and st.session_state.category == "HVAC":
    PDF_URL = "https://raw.githubusercontent.com/Maniyuvi/CSvFile/main/om_pead-rp71-140jaa_kd79d904h01%20(1).pdf"
elif "category" in st.session_state and st.session_state.category == "CCTV System":
    PDF_URL = "https://raw.githubusercontent.com/Maniyuvi/CSvFile/main/nmc110.pdf"


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

st.set_page_config(page_title=st.session_state.header_name, layout="centered", page_icon="etihad_logo.png" if st.session_state.header_name == "Etihad Rail" else "bot.png")
if st.session_state.header_name == "Etihad Rail":    
    bot_icon = img_to_base64("etihad_logo.png")
else:
    bot_icon = img_to_base64("bot.png")

def toggle_header():
    if st.session_state.header_name == "Etihad Rail":
        st.session_state.header_name = "MaintainX AI"
    else:
        st.session_state.header_name = "Etihad Rail"

def toggle_upload_chat():
    st.session_state.gemini_upload = not st.session_state.get("gemini_upload", False)


# ---- Enhanced CSS with better mobile support ----
st.markdown(f"""
    <style>
    /* Use CSS variables that adapt to Streamlit's theme */
    :root {{
        --border-color: #e0e0e0;
        --shadow: rgba(0,0,0,0.1);
    }}
    .fixed-title {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        border-bottom: 4px solid var(--border-color);
        height: 64px;
        z-index: 999;
    }}

    .block-container {{
        max-width: 100% !important;
        padding-left: 30px;
        padding-right: 30px;
        padding-top: 30px;
        padding-bottom: 30px;
    }}
    
    [data-testid="stHorizontalBlock"] .stColumn:nth-child(2) [data-testid="stVerticalBlock"] {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        height: calc(100vh - 140px);
        min-height: 320px;
        max-height: 500px;
        box-sizing: border-box;
    }}  
              
    [data-testid="stHorizontalBlock"] .stColumn:nth-child(2) [data-testid="stLayoutWrapper"]:nth-child(1) {{
        order: 1;
        flex: 1 1 auto;
        min-height: 0;
        overflow-y: auto;
    }}
            
    [data-testid="stHorizontalBlock"] .stColumn:nth-child(2) [data-testid="stLayoutWrapper"]:has([data-testid="stExpander"]) {{
        order: 2;
        flex: none;
        margin-top: 6px;
        max-height: 200px;
        overflow-y: auto;
    }}
            
    [data-testid="stHorizontalBlock"] .stColumn:nth-child(2) [data-testid="stElementContainer"]:has([data-testid="stChatInput"]) {{
        order: 3;
        position: relative;
        bottom: 0;
    }}

    [data-testid="stHorizontalBlock"] .stColumn:nth-child(2) [data-testid="stVerticalBlock"]:has([data-testid="stLayoutWrapper"]) {{
        border: 1px solid #e0e0e0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-radius: 15px;
        padding: 10px;
    }}
    [data-testid="stHorizontalBlock"] > .stColumn:nth-child(1) [data-testid="stVerticalBlock"] [data-testid="stLayoutWrapper"] {{
        height: 35px;
    }}
    .custom-chat-input-wrapper + [data-testid="stElementContainer"] {{
        padding: 50px;
    }}
    .user-message {{
        display: flex;
        justify-content: flex-end;
        text-align: right;
        margin-bottom: 12px;
    }}

    header[data-testid="stHeader"]::after {{
        content: "{st.session_state.header_name}";
        font-size: 2.5rem;
        font-weight: 650;
        color: var(--text-color);
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
    }}
    
    .bot-message {{
        display: flex;
        justify-content: flex-start;
        text-align: left;
        margin-bottom: 12px;
    }}
    
    .chat-bubble {{
        max-width: 75%;
        padding: 14px 18px;
        border-radius: 18px;
        margin: 4px 8px;
        word-wrap: break-word;
        font-size: 1rem;
        line-height: 1.4;
        box-shadow: 0 2px 10px var(--shadow);
    }}
    
    .user-bubble {{
        background: linear-gradient(135deg, #007aff, #0051d5);
        color: white;
        border-bottom-right-radius: 6px;
    }}
    
    .bot-bubble {{
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 6px;
    }}
    
    .stats-container {{
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        color: var(--text-primary);
    }}
    
    .upload-progress {{
        background: var(--bg-secondary);
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: var(--text-primary);
    }}
    
    .success-message {{
        background: var(--bg-secondary);
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #4caf50;
    }}
    
    .warning-message {{
        background: var(--bg-secondary);
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #ff9800;
    }}
    
    /* Fix Streamlit's default dark mode text colors */
    [data-theme="dark"] .stats-container h4,
    [data-theme="dark"] .stats-container p {{
        color: var(--text-primary) !important;
    }}

    /* Fixed FAQ Container styling */
    .faq-container {{
        position: sticky;
        bottom: 0;
        background: var(--bg-secondary);
        border-top: 1px solid var(--border-color);
        padding: 12px 0;
        z-index: 100;
        box-shadow: 0 -2px 10px var(--shadow);
        margin-top: 10px;
    }}

    .faq-button {{
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
    }}

    .faq-button:hover {{
        background: linear-gradient(135deg, #0051d5, #003d99);
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
    }}

    .faq-content {{
        margin-top: 12px;
        padding: 12px;
        background: var(--bg-primary);
        border-radius: 8px;
        border-left: 4px solid #007aff;
    }}
            
    /* Base layout for messages */
    .user-message,
    .bot-message {{
        display: flex;
        align-items: center;
    }}

    /* User messages → right aligned */
    .user-message {{
        justify-content: flex-end;
    }}

    /* Bot messages → left aligned */
    .bot-message {{
        justify-content: flex-start;
    }}

    /* User icon */
    .user-message::before {{
        content: "";
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-image: url("https://cdn-icons-png.flaticon.com/512/847/847969.png");
        background-size: cover;
        background-position: center;
        flex-shrink: 0;
    }}

    /* Bot icon */
    .bot-message::before {{
        content: "";
        width: 42px;
        height: 32px;
        background-image: url("data:image/png;base64,{bot_icon}");
        background-size: cover;
        background-position: center;
        flex-shrink: 0;
    }}

    /* Mobile responsiveness */
    @media screen and (max-width: 768px) {{
        header[data-testid="stHeader"]::after {{
            font-size: 2.0rem !important;
            font-weight: 600 !important;
        }}

        h2 {{
            font-size: 1.5rem !important;
        }}

        .chat-bubble {{
            font-size: 0.9rem !important;
            padding: 10px 14px !important;
            max-width: 85% !important;
        }}

        .faq-button {{
            font-size: 0.85rem !important;
            padding: 8px 12px !important;
        }}

        .stats-container, .upload-progress, .success-message, .warning-message {{
            font-size: 0.85rem !important;
            padding: 8px !important;
        }}

        .block-container {{
            padding-left: 15px !important;
            padding-right: 15px !important;
        }}

        [data-testid="stHorizontalBlock"] .stColumn:nth-child(2) [data-testid="stVerticalBlock"] {{
            height: calc(100vh - 180px) !important;
            max-height: 450px !important;
        }}
    }}

    </style>
""", unsafe_allow_html=True)

if "verification_chat_open" not in st.session_state:
    st.session_state.verification_chat_open = False
if "selected_checklist" not in st.session_state:
    st.session_state.selected_checklist = None

def render_chat_assistant(instance="default"):

    # Namespaced keys (unique per instance)
    chat_key = f"chat_history_{instance}"
    faq_key = f"faq_open_{instance}"
    rerank_key = f"rerank_{instance}"

    # Ensure defaults exist
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []  # list of {role, content, groundings?}

    if faq_key not in st.session_state:
        st.session_state[faq_key] = False

    # If you have a rerank toggle elsewhere, keep it per-instance too
    if rerank_key not in st.session_state:
        st.session_state[rerank_key] = False

    chat_history = st.session_state[chat_key]

    if total_vectors == 0:
        st.warning("⚠️ No documents found. Please upload some PDF documents first.")
        if st.button("📤 Upload PDFs", key=f"upload_btn_{instance}", type="primary"):
            st.session_state.page = "Upload PDFs"  # if you use this global page state
            st.rerun()

    else:
        # Chat interface container
        chat_container = st.container()

        with chat_container:
            for i, msg in enumerate(chat_history):
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

                # Show source button per-instance and per-message (unique key)
                if msg["role"] == "assistant" and msg.get("groundings"):
                    grounding = msg["groundings"][0]
                    src = grounding.get("source")
                    page_no = grounding.get("page_number")

                    if src and page_no:
                        if st.button("📄 View Source", key=f"view_source_{instance}_{i}"):
                            src_norm = normalize(src)
                            url = next(u for u in URL_LIST if normalize(os.path.basename(u)) == src_norm)
                            png_bytes = render_pdf_page_to_png_bytes(url, page_number=int(page_no), zoom=2.0)
                            show_source_dialog(png_bytes)
                if msg["role"] == "assistant" and msg.get("audio_byte"):
                    st.audio(io.BytesIO(msg["audio_byte"]), autoplay=True)
            
            # Auto-scroll anchor at the end of messages
            if chat_history:
                # Create an anchor element at the bottom of chat
                st.markdown(f'<div id="chat-bottom-anchor-{instance}"></div>', unsafe_allow_html=True)
                
                # Use st.components.v1.html to inject JavaScript that actually executes
                import streamlit.components.v1 as components
                components.html(f"""
                    <script>
                        // Wait for DOM to be ready then scroll within container only
                        setTimeout(function() {{
                            // Find the last bot message
                            const botMessages = parent.document.querySelectorAll('.bot-message');
                            if (botMessages.length > 0) {{
                                const lastBotMessage = botMessages[botMessages.length - 1];
                                
                                // Find the scrollable parent container (not the page itself)
                                let scrollContainer = lastBotMessage.parentElement;
                                while (scrollContainer) {{
                                    const style = parent.window.getComputedStyle(scrollContainer);
                                    const overflowY = style.getPropertyValue('overflow-y');
                                    if (overflowY === 'auto' || overflowY === 'scroll') {{
                                        // Calculate the position to scroll the last bot message to the top
                                        const containerRect = scrollContainer.getBoundingClientRect();
                                        const messageRect = lastBotMessage.getBoundingClientRect();
                                        const scrollOffset = messageRect.top - containerRect.top + scrollContainer.scrollTop;
                                        
                                        // Scroll to position the last bot message at the top with some padding
                                        scrollContainer.scrollTo({{
                                            top: scrollOffset - 10,
                                            behavior: 'smooth'
                                        }});
                                        break;
                                    }}
                                    // Stop before reaching the main page scroll
                                    if (scrollContainer.tagName === 'MAIN' || scrollContainer.tagName === 'BODY') {{
                                        break;
                                    }}
                                    scrollContainer = scrollContainer.parentElement;
                                }}
                            }}
                        }}, 150);
                    </script>
                """, height=0)

        # Chat input - keep the same UI but append to namespaced history
        if instance == "side" and st.session_state.verification_chat_open and st.session_state.get("selected_checklist", None):
            user_input = st.session_state["selected_checklist"]
            st.session_state["selected_checklist"] = None
            
            # Mobile auto-scroll: inject scroll script when a new Ask Agent button is clicked
            if st.session_state.get('mobile_scroll_pending', False):
                import streamlit.components.v1 as components
                # Use st.sidebar or a separate container to avoid layout disruption
                components.html("""
                    <script>
                        setTimeout(function() {
                            if (window.parent.innerWidth <= 640) {
                                const columns = parent.document.querySelectorAll('[data-testid="stHorizontalBlock"] > .stColumn');
                                if (columns.length >= 2) {
                                    const chatColumn = columns[1];
                                    if (chatColumn) {
                                        chatColumn.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                    }
                                }
                            }
                        }, 200);
                    </script>
                """, height=0)
                st.session_state.mobile_scroll_pending = False
        else:
            user_input = st.chat_input("Ask about your documents...",accept_audio=True)

        if user_input:
            
            try:
                # Determine if we have audio or text input
                has_audio = not isinstance(user_input, str) and getattr(user_input, 'audio', None)
                
                if has_audio:
                    audio_file = user_input.audio
                    user_query = transcribe_audio(audio_file)
                    
                    # Add user message to the namespaced history
                    st.session_state[chat_key].append({"role": "user", "content": user_query if user_query.strip() != "" else " "})
                    
                    # Process query
                    with st.spinner("🔍 Searching your documents..."):
                        bot_reply, source = process_user_query(
                            user_query,
                            st.session_state[chat_key][:-1],
                            rerank=st.session_state.get(rerank_key, False),
                            category=st.session_state.get("category", None),
                            type=st.session_state.get("type", None),
                            brand=st.session_state.get("brand", None),
                            model_series=st.session_state.get("model_series", None),
                            is_side = True if instance == "side" else False 
                        )
                        audio_byte = generate_audio_response(bot_reply)
                else:
                    # Handle text input (either string from checklist or object.text from chat_input)
                    query_text = user_input if isinstance(user_input, str) else user_input.text.strip()
                    
                    # Add user message to the namespaced history
                    st.session_state[chat_key].append({"role": "user", "content": query_text})

                     # Process query
                    with st.spinner("🔍 Searching your documents..."):
                        bot_reply, source = process_user_query(
                            query_text,
                            st.session_state[chat_key][:-1],  # previous messages for context
                            rerank=st.session_state.get(rerank_key, False),
                            category=st.session_state.get("category", None),
                            type=st.session_state.get("type", None),
                            brand=st.session_state.get("brand", None),
                            model_series=st.session_state.get("model_series", None),
                            is_side = True if instance == "side" else False 
                        )
                        audio_byte = None
                # Prepare grounding metadata list from matches
                groundings = []
                if source.get('source') and source.get('page'):
                    groundings.append({
                        'source': source.get('source'),
                        'page_number': source.get('page')
                    })

                # Add bot response to namespaced history
                if audio_byte:
                    st.session_state[chat_key].append({"role": "assistant", "content": bot_reply, "groundings": groundings, "audio_byte": audio_byte.getvalue()})
                else:
                    st.session_state[chat_key].append({"role": "assistant", "content": bot_reply, "groundings": groundings})

            except Exception as ex:
                error_msg = f"Sorry, I encountered an error: {str(ex)}"
                st.session_state[chat_key].append({"role": "assistant", "content": error_msg})
                st.error(f"Error processing query: {ex}")

            st.rerun()

        # Frequently Asked Questions / Suggestions (expander - uses instance-specific state)
        with st.expander("💡 Frequently Asked Questions / Suggestions", expanded=st.session_state[faq_key]):
            if st.session_state.get("category") == "HVAC":
                suggested_questions = [
                    "What are the key safety procedures described in the documents?",
                    "Summarize maintenance schedule guidelines.",
                    "What are the contact details for emergency?",
                    "Available temperature ranges?"
                ]
            elif st.session_state.get("category") == "CCTV System":
                suggested_questions = [
                        "Why is the camera not powering ON?",
                        "Why is the camera not accessible on the network?",
                        "Why is live video not displaying or freezing?",
                        "Why are motion detection or alarm events not triggering?"]
            else:
                suggested_questions = [
                    "What are the key safety procedures described in the documents?",
                    "Summarize maintenance schedule guidelines.",
                    "What are the contact details for emergency?",
                    "Available temperature ranges?"
                ]

            # Display questions in a single column layout
            for i, q in enumerate(suggested_questions):
                if st.button(q, key=f"suggestion_{instance}_{i}", use_container_width=True):
                    # When a suggestion is clicked, add as user input and process it
                    st.session_state[chat_key].append({"role": "user", "content": q})
                    try:
                        with st.spinner("🔍 Searching your documents..."):
                            bot_reply, source = process_user_query(
                                q,
                                st.session_state[chat_key][:-1],
                                rerank=st.session_state.get(rerank_key, False),
                                category=st.session_state.get("category", None) if instance == "side" else None,
                                type=st.session_state.get("type", None) if instance == "side" else None,
                                brand=st.session_state.get("brand", None) if instance == "side" else None,
                                model_series=st.session_state.get("model_series", None) if instance == "side" else None,
                                is_side = True if instance == "side" else False 
                            )

                            groundings = []
                            if source.get('source') and source.get('page'):
                                groundings.append({
                                    'source': source.get('source'),
                                    'page_number': source.get('page')
                                })

                            st.session_state[chat_key].append({"role": "assistant", "content": bot_reply, "groundings": groundings})
                    except Exception as ex:
                        error_msg = f"Sorry, I encountered an error: {str(ex)}"
                        st.session_state[chat_key].append({"role": "assistant", "content": error_msg})
                        st.error(f"Error processing suggestion: {ex}")

                    # Auto-close the FAQ and rerun to show updated chat
                    st.session_state[faq_key] = False
                    st.rerun()

@st.dialog("Source page", width="medium")
def show_source_dialog(png_bytes: bytes):
    st.image(png_bytes)

# ---- Fixed Title ----
st.markdown('<div class="fixed-title"></div>', unsafe_allow_html=True)
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
        
        if total_vectors == 0:
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
        default_page = "Category Selection"
    
    # on = st.toggle("Enable Reranking", value=False, help="Toggle to enable or disable reranking of search results")
    # if on:
    #     st.session_state.rerank = True
    # else:
    #     st.session_state.rerank = False
    
    page = st.radio(
        "Choose Action:",
        options=["Category Selection", "Checklist", "Chat Assistant", "Upload PDFs", "Database Management"],
        index=["Category Selection", "Checklist", "Chat Assistant", "Upload PDFs", "Database Management"].index(default_page),
        key = "page"
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
                        pinecone_index_name,
                        use_gemini=st.session_state.get("gemini_upload", False)
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
    render_chat_assistant(instance="main")

# ---- Database Management Page ----
elif page == "Database Management":
    st.header("Database Management")
    
    # Show detailed statistics
    try:
        stats = get_index_stats(pinecone_api_key, pinecone_index_name)
        
        # Show recently processed files
        if st.session_state.processed_files:
            st.subheader("Recently Processed Files")
            
            for file_info in st.session_state.processed_files[-20:]:  # Show last 20
                status_icon = "✅" if file_info['status'] == 'success' else "❌"
                error_text = f" ({file_info.get('error', 'Unknown error')})" if file_info['status'] == 'error' else ""
                st.markdown(f"{status_icon} {file_info['name']}{error_text}")
        
    except Exception as e:
        st.error(f"Error retrieving database statistics: {e}")
    
    # Database operations
    st.subheader("Clear Chat History")
    
    # Clear chat history
    if st.button("Clear", help="Clear conversation history (keeps documents)"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    # Reset entire database
    st.subheader("Reset Database")
    
    reset_confirm = st.checkbox("I understand this will delete all documents")
    
    if reset_confirm:
        st.warning("**Warning:** This will delete all uploaded documents and cannot be undone.")
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
    st.subheader("Change Header")
    st.button("Change", on_click=toggle_header)

    st.subheader("Change Upload Process")
    st.button("Switch", on_click=toggle_upload_chat, key="change_upload_btn")

    if st.session_state.get("gemini_upload", False):
        st.markdown("Gemini Upload Enabled")
    else:
        st.markdown("OpenAI Upload Enabled")

elif page == "Category Selection":
    st.session_state.chat_history_side = []
    st.session_state.verification_chat_open = False
    st.header("📂 Category Selection")

    df = pd.read_excel('Model_Series.xlsx')

    df['Model / Series'] = df['Model / Series'].str.split('; ')
    master_df = df.explode('Model / Series').reset_index(drop=True)
    master_df['Model / Series'] = master_df['Model / Series'].str.strip()

    # A. Picklist 1 (Category)
    category_options = master_df['Category'].unique()
    selected_category = st.selectbox(
        'Category',
        category_options
    )
    st.session_state.category = selected_category

    # Filter DF based on Category selection
    df_filtered_by_category = master_df[master_df['Category'] == selected_category]


    # B. Picklist 2 (Type)
    type_options = df_filtered_by_category['Type'].unique()
    selected_type = st.selectbox(
        'Type',
        type_options
    )
    st.session_state.type = selected_type

    # Filter DF based on Model selection (and Category selection is still active)
    df_filtered_by_type = df_filtered_by_category[df_filtered_by_category['Type'] == selected_type]


    # C. Picklist 3 (Brand)
    brand_options = df_filtered_by_type['Brand'].unique()
    selected_brand = st.selectbox(
        'Brand',
        brand_options
    )
    st.session_state.brand = selected_brand

    # Filter DF based on Brand selection (and previous selections are still active)
    df_filtered_by_brand = df_filtered_by_type[df_filtered_by_type['Brand'] == selected_brand]

    # D. Picklist 4 (Model / Series)
    model_series_options = df_filtered_by_brand['Model / Series'].unique()
    selected_model_series = st.selectbox(
        'Model / Series',
        model_series_options
    )
    st.session_state.model_series = selected_model_series

    def go_to_checklist():
        st.session_state.page = "Checklist"

    st.button("Confirm", on_click=go_to_checklist)

#---- Checklist Page ----
elif page == "Checklist":

    st.header(f"✅ Checklist - {st.session_state.get('brand', '')} {st.session_state.get('model_series', '')}")
    col_main, col_chat = st.columns([2, 1])

    with col_main:
        
        def on_arrow_click(text: str):
            st.session_state.selected_checklist = text
            st.session_state.verification_chat_open = True
            st.session_state.mobile_scroll_pending = True

        if st.session_state.get('category') == "HVAC":
            CHECKS = [
                ("chk_temp_modes", "Check that the temperature is set correctly for Cooling, Heating, and Auto modes."),
                ("chk_on_lamp", "Check if the ON lamp on the wired controller is flashing and record the error code."),
                ("chk_wireless_lamp", "Check if the lamp near the wireless receiver on the indoor unit is flashing?."),
                ("chk_mode", "Check that the correct operating mode (Cool / Heat / Dry / Fan / Auto / Vent) is selected."),
                ("chk_remote_error", "Check if any error code is shown on the remote display."),
                ("chk_timer", "Check that the timer settings are set correctly and only one timer type is in use."),
                ("chk_filters", "Check that the air filters are clean, in good condition, and fitted properly."),
                ("chk_alarm", "Check if any alarm or flashing light is present and record the details."),
            ]
            for chk_key, label in CHECKS:
                st.checkbox(label, key=chk_key)
                st.button(
                    "Ask Agent",
                    key=f"btn_{chk_key}",
                    on_click=on_arrow_click,
                    args=(label,),
                    type='tertiary'
                )
        else:
            CHECKS = [
                ("chk_power", "Check that the camera power supply is stable at DC 12V and the unit powers ON correctly."),
                ("chk_network", "Check if the network link LED is active and the camera IP is reachable on the network."),
                ("chk_video", "Check if live video is displayed correctly in the web browser without freezing or delay."),
                ("chk_video_settings", "Check that the correct video resolution, frame rate, and compression settings are applied."),
                ("chk_lens", "Check the camera lens for dust or damage and confirm image focus and clarity."),
                ("chk_motion", "Check if motion detection is enabled and verify correct detection response."),
                ("chk_alarm_io", "Check alarm input and output connections and confirm correct NO/NC operation."),
                ("chk_time", "Check system date and time settings and confirm synchronization is correct."),
            ]
            for chk_key, label in CHECKS:
                st.checkbox(label, key=chk_key)
                st.button(
                    "Ask Agent",
                    key=f"btn_{chk_key}",
                    on_click=on_arrow_click,
                    args=(label,),
                    type='tertiary'
                )
    with col_chat:
        if st.session_state.verification_chat_open:
            render_chat_assistant(instance="side")
    
    bottom_bar = st.container(horizontal=True, horizontal_alignment="right")
    with bottom_bar:
        if st.session_state.verification_chat_open:
            if st.button("Close Agent", type="primary"):
                st.session_state.verification_chat_open = False
                st.rerun()
        # else:
        #     if st.button("Close Agent", type="secondary"):
        #         st.session_state.verification_chat_open = False
        #         st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
