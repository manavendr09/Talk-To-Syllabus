import streamlit as st
import asyncio
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
import tempfile
import os
import zipfile
import json
import requests
from bs4 import BeautifulSoup
import markdown
from urllib.parse import urlparse
import numpy as np
import pickle


# =========================
# 🔹 Simple Vector Store (No FAISS dependency)
# =========================
class SimpleVectorStore:
    """A simple vector store that doesn't require FAISS - works on Python 3.13+"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.embeddings_list = []
    
    def add_documents(self, documents):
        """Add documents to the store"""
        if not documents:
            return
        
        self.documents.extend(documents)
        
        # Create embeddings for each document
        texts = [doc.page_content for doc in documents]
        new_embeddings = self.embeddings.embed_documents(texts)
        self.embeddings_list.extend(new_embeddings)
    
    def similarity_search(self, query, k=2):
        """Search for similar documents using cosine similarity"""
        if not self.documents:
            return []
        
        try:
            # Embed the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate cosine similarity
            similarities = []
            for doc_embedding in self.embeddings_list:
                # Convert to numpy arrays
                q = np.array(query_embedding)
                d = np.array(doc_embedding)
                
                # Cosine similarity
                norm_q = np.linalg.norm(q)
                norm_d = np.linalg.norm(d)
                
                if norm_q == 0 or norm_d == 0:
                    similarity = 0
                else:
                    similarity = np.dot(q, d) / (norm_q * norm_d)
                
                similarities.append(similarity)
            
            # Get top k indices
            k = min(k, len(similarities))
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            # Return top k documents
            return [self.documents[i] for i in top_indices]
        except Exception as e:
            print(f"Error in similarity_search: {str(e)}")
            return []
    
    def __len__(self):
        return len(self.documents)
    
    @property
    def index(self):
        """Mock index property for compatibility with existing code"""
        class MockIndex:
            def __init__(self, parent):
                self.parent = parent
            @property
            def ntotal(self):
                return len(self.parent)
        
        return MockIndex(self)


# Initialize session state first
def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "processed_files": set(),
        "pdf_enabled": True,
        "notion_enabled": False,
        "wiki_enabled": False,
        "show_sources": True,
        "uploaded_content": [],
        "wiki_url_value": ""  # For wiki URL input
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize session state
initialize_session_state()

# =========================
# 🔹 API Key
# =========================
GOOGLE_API_KEY = "your api key"  # Replace with your actual API key or use environment variable

# =========================
# 🔹 Page Config & Styling
# =========================
st.set_page_config(
    page_title="Talk To Syllabus - An Intelligent Q&A Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #0f172a 0%, #1e293b 50%, #111827 100%) !important;
            color: #ffffff !important;
        }
        .user-bubble {
            background: linear-gradient(135deg, #2563eb, #1e40af);
            color: white;
            padding: 12px 18px;
            border-radius: 20px;
            margin: 8px 0;
            max-width: 70%;
            align-self: flex-end;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }
        .bot-bubble {
            background: linear-gradient(135deg, #f97316, #ea580c);
            color: white;
            padding: 12px 18px;
            border-radius: 20px;
            margin: 8px 0;
            max-width: 70%;
            align-self: flex-start;
            box-shadow: 0 4px 15px rgba(249, 115, 22, 0.3);
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            padding: 1rem;
        }
        .sidebar-section {
            padding: 1rem 0;
            border-bottom: 1px solid #444;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #ffffff !important;
        }
        .content-item {
            background: rgba(255, 255, 255, 0.08);
            padding: 8px;
            margin: 4px 0;
            border-radius: 6px;
            font-size: 0.9rem;
            color: #ffffff !important;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.05);
            color: #ffffff !important;
        }
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .gradient-text {
            background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        .stButton > button {
            background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 26px rgba(124,58,237,0.30);
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🔹 Event loop fix
# =========================
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# =========================
# 🔹 Content Processing Functions
# =========================
def process_pdf_file(file):
    try:
        pdf_reader = PdfReader(file)
        documents = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""

            if text.strip():
                chunks = splitter.split_text(text)

                for i, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file.name,
                                "source_file": file.name,
                                "page_number": page_num + 1,
                                "chunk_id": i + 1,
                                "type": "PDF"
                            }
                        )
                    )

        return documents, None

    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"


def process_notion_export(file):
    """Process Notion export zip file - returns Document objects"""
    try:
        documents = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        with zipfile.ZipFile(file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.md'):
                    with zip_ref.open(file_name) as md_file:
                        content = md_file.read().decode('utf-8')
                        # Convert markdown to plain text
                        html = markdown.markdown(content)
                        soup = BeautifulSoup(html, 'html.parser')
                        text = soup.get_text()

                        if text.strip():
                            chunks = splitter.split_text(text)
                            for i, chunk in enumerate(chunks):
                                documents.append(
                                    Document(
                                        page_content=chunk,
                                        metadata={
                                            "source": file.name,
                                            "source_file": file.name,
                                            "notion_page": file_name,
                                            "chunk_id": i + 1,
                                            "type": "Notion"
                                        }
                                    )
                                )

        return documents, None if documents else "No readable content found in Notion export."

    except Exception as e:
        return None, f"Error processing Notion export: {str(e)}"


def process_wiki_url(url):
    """Process Wikipedia or other wiki URL - returns Document objects"""
    try:
        headers = {'User-Agent': 'TalkToSyllabus/1.0 (Educational Use)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract main content
        content_selectors = ['#mw-content-text', '.mw-parser-output', 'main', 'article']
        text = ""

        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator=' ', strip=True)
                break

        if not text:
            text = soup.get_text(separator=' ', strip=True)

        if not text.strip():
            return None, "No readable content found at the URL."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)
        documents = []
        parsed_url = urlparse(url)
        page_name = parsed_url.path.split('/')[-1] or parsed_url.netloc

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "source_file": page_name,
                        "url": url,
                        "chunk_id": i + 1,
                        "type": "Wiki"
                    }
                )
            )

        return documents, None

    except Exception as e:
        return None, f"Error processing URL: {str(e)}"


# =========================
# 🔹 Vector Store Functions (Uses SimpleVectorStore - No FAISS)
# =========================

def create_vector_store(documents):
    """Create a new vector store using SimpleVectorStore"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        vector_store = SimpleVectorStore(embeddings)
        vector_store.add_documents(documents)
        return vector_store, None

    except Exception as e:
        return None, f"Error creating vector store: {str(e)}"


def update_vector_store(new_documents):
    """Update the vector store with new documents"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        if st.session_state.vector_store is None:
            st.session_state.vector_store = SimpleVectorStore(embeddings)
            st.session_state.vector_store.add_documents(new_documents)
        else:
            st.session_state.vector_store.add_documents(new_documents)

        return True, None

    except Exception as e:
        return False, f"Error updating vector store: {str(e)}"


# =========================
# 🔹 Answer Generation (Unchanged - Your exact model)
# =========================
def get_answer_simple(user_query, vector_store):
    """Get answer using simple approach"""
    try:
        matching_chunks = vector_store.similarity_search(user_query, k=2)

        if not matching_chunks:
            return "I don't know Manavendra", []

        context = "\n\n".join([chunk.page_content for chunk in matching_chunks])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            max_output_tokens=1024,
            google_api_key=GOOGLE_API_KEY
        )

        prompt = f"""
        You are an assistant tutor.

        Answer the question ONLY using the given context.

        Context:
        {context}

        Question:
        {user_query}

        Answer:
        """

        response = llm.invoke(prompt)

        # Clean extraction
        if isinstance(response.content, list):
            output = response.content[0]["text"]
        else:
            output = response.content

        sources = []
        for chunk in matching_chunks:
            sources.append({
                "content": chunk.page_content[:200] + "...",
                "metadata": chunk.metadata
            })

        return output, sources

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}", []


# =========================
# 🔹 Sidebar
# =========================
def render_sidebar():
    """Render the sidebar with multiple upload options"""
    with st.sidebar:
        st.markdown("## 📁 Upload Content")
        st.markdown("---")

        # PDF Files Section
        with st.expander("📄 PDF Files", expanded=True):
            st.markdown("""
            <div class="upload-area">
                <p>📄 Drag and drop PDF files here</p>
                <p><small>Limit 200MB per file</small></p>
            </div>
            """, unsafe_allow_html=True)

            pdf_files = st.file_uploader(
                "Browse PDF files",
                type="pdf",
                accept_multiple_files=True,
                key="pdf_uploader",
                label_visibility="collapsed"
            )

            if pdf_files:
                for pdf_file in pdf_files:
                    if pdf_file.name not in [item['name'] for item in st.session_state.uploaded_content if
                                             item['type'] == 'PDF']:
                        with st.spinner(f"Processing {pdf_file.name}..."):
                            documents, error = process_pdf_file(pdf_file)

                            if documents:
                                success, vector_error = update_vector_store(documents)
                                if success:
                                    st.session_state.uploaded_content.append({
                                        'name': pdf_file.name,
                                        'type': 'PDF',
                                        'chunks': len(documents)
                                    })
                                    st.success(f"✅ {pdf_file.name} processed!")
                                else:
                                    st.error(f"Vector store error: {vector_error}")
                            else:
                                st.error(f"Error: {error}")

        # Notion Exports Section
        with st.expander("📝 Notion Exports", expanded=False):
            notion_file = st.file_uploader(
                "Upload Notion export (ZIP file)",
                type="zip",
                key="notion_uploader",
                label_visibility="collapsed"
            )

            if notion_file:
                if notion_file.name not in [item['name'] for item in st.session_state.uploaded_content if
                                            item['type'] == 'Notion']:
                    with st.spinner(f"Processing {notion_file.name}..."):
                        documents, error = process_notion_export(notion_file)
                        if documents:
                            success, vector_error = update_vector_store(documents)
                            if success:
                                st.session_state.uploaded_content.append({
                                    'name': notion_file.name,
                                    'type': 'Notion',
                                    'chunks': len(documents)
                                })
                                st.success(f"✅ {notion_file.name} processed!")
                            else:
                                st.error(f"Vector store error: {vector_error}")
                        else:
                            st.error(f"Error: {error}")

        # Wiki Pages Section
        with st.expander("🌐 Wiki Pages", expanded=False):
            # Use a separate session state variable for the wiki URL
            if "wiki_url_temp" not in st.session_state:
                st.session_state.wiki_url_temp = ""
                
            wiki_url = st.text_input(
                "Enter Wikipedia or wiki URL:", 
                key="wiki_url", 
                placeholder="https://...",
                value=st.session_state.wiki_url_temp
            )

            col1, col2 = st.columns([3, 1])
            with col2:
                add_wiki_button = st.button("➕ Add", key="add_wiki", use_container_width=True)

            if add_wiki_button and wiki_url:
                parsed_url = urlparse(wiki_url)
                page_name = parsed_url.path.split('/')[-1] or parsed_url.netloc

                if page_name not in [item['name'] for item in st.session_state.uploaded_content if
                                     item['type'] == 'Wiki']:
                    with st.spinner(f"Processing {page_name}..."):
                        documents, error = process_wiki_url(wiki_url)
                        if documents:
                            success, vector_error = update_vector_store(documents)
                            if success:
                                st.session_state.uploaded_content.append({
                                    'name': page_name,
                                    'type': 'Wiki',
                                    'chunks': len(documents),
                                    'url': wiki_url
                                })
                                st.success(f"✅ {page_name} processed!")
                                # Clear the input
                                st.session_state.wiki_url_temp = ""
                                st.rerun()
                            else:
                                st.error(f"Vector store error: {vector_error}")
                        else:
                            st.error(f"Error: {error}")
                else:
                    st.warning("This page has already been added!")
            elif add_wiki_button and not wiki_url:
                st.warning("Please enter a valid URL!")

        # Show Sources Toggle
        st.markdown("---")
        st.session_state.show_sources = st.checkbox("🔍 Show Sources", value=st.session_state.show_sources)

        # Uploaded Content Summary
        if st.session_state.uploaded_content:
            st.markdown("---")
            st.markdown("### 📚 Uploaded Content")
            for item in st.session_state.uploaded_content:
                icon = {"PDF": "📄", "Notion": "📝", "Wiki": "🌐"}.get(item['type'], "📄")
                st.markdown(f"""
                <div class="content-item">
                    {icon} <strong>{item['name']}</strong><br>
                    <small>{item['type']} • {item['chunks']} chunks</small>
                </div>
                """, unsafe_allow_html=True)

            # Vector store info
            if st.session_state.vector_store:
                st.markdown(f"*Total vectors: {len(st.session_state.vector_store)}*")

        # Control Buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button("🗑 Clear All", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.vector_store = None
                st.session_state.uploaded_content = []
                st.success("All content cleared!")
                st.rerun()


# =========================
# 🔹 Main Application
# =========================
def main():
    """Main application"""
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 class="gradient-text" style="font-size: 2.5rem;">
                📘 Talk To Syllabus - An Intelligent Q&A Assistant
            </h1>
            <p style="color: #e2e8f0; font-size: 1.1rem;">
                Chat with your PDFs, Notion exports, and Wiki pages using AI-powered intelligence
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Render the sidebar
    render_sidebar()

    # Main chat interface
    if st.session_state.uploaded_content and st.session_state.vector_store:
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                st.markdown(
                    f"<div class='chat-container'><div class='user-bubble'>{chat['content']}</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-container'><div class='bot-bubble'>{chat['content']}</div></div>",
                    unsafe_allow_html=True
                )

                # Show sources for historical messages if available and enabled
                if st.session_state.show_sources and 'sources' in chat and chat[
                    'sources'] and "I don't know Manavendra" not in chat['content']:
                    with st.expander(f"📚 Sources for message {(i // 2) + 1}", expanded=False):
                        for j, source in enumerate(chat['sources']):
                            metadata = source.get('metadata', {})
                            source_file = metadata.get('source_file', metadata.get('source', 'Unknown'))
                            source_type = metadata.get('type', 'Document')

                            # Create header based on source type
                            if source_type == 'PDF':
                                page_number = metadata.get('page_number', 'N/A')
                                header = f"📄 *{source_file}* - Page {page_number}"
                            elif source_type == 'Notion':
                                notion_page = metadata.get('notion_page', '')
                                header = f"📝 *{source_file}* - {notion_page}"
                            elif source_type == 'Wiki':
                                header = f"🌐 *{source_file}*"
                            else:
                                header = f"📄 *{source_file}*"

                            st.markdown(header)
                            st.text_area(
                                f"source_{i}_{j}",
                                value=source['content'],
                                height=120,
                                disabled=True,
                                label_visibility="collapsed"
                            )

                            # Add URL if available (for wiki sources)
                            if 'url' in metadata:
                                st.markdown(f"🔗 [View Original]({metadata['url']})")

                            st.markdown("---")

        # User query input
        user_query = st.chat_input("💬 Ask a question about your uploaded content...")

        if user_query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.spinner("🤔 Thinking..."):
                output, sources = get_answer_simple(user_query, st.session_state.vector_store)

            # Add bot response to chat history with sources
            bot_message = {"role": "bot", "content": output}
            if sources:
                bot_message["sources"] = sources
            st.session_state.chat_history.append(bot_message)

            st.rerun()

    elif not st.session_state.uploaded_content:
        # Welcome message with instructions
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 3rem 1rem; background: rgba(255,255,255,0.03); border-radius: 15px;">
                    <h2 style="color: #e2e8f0;">👋 Welcome to Talk To Syllabus!</h2>
                    <p style="color: #94a3b8; font-size: 1.1rem; margin: 1.5rem 0;">
                        Get started by uploading your content using the sidebar.
                    </p>
                    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 2rem;">📄</div>
                            <p style="color: #e2e8f0;">PDF Files</p>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem;">📝</div>
                            <p style="color: #e2e8f0;">Notion Exports</p>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2rem;">🌐</div>
                            <p style="color: #e2e8f0;">Wiki Pages</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.error("There was an issue processing your content. Please try uploading again.")


# =========================
# 🔹 Run Main App
# =========================
if __name__ == "__main__":
    main()
