# 📘 Talk To Syllabus - An Intelligent Q&A Assistant

An AI-powered chatbot that allows you to chat with your PDFs, Notion exports, and Wikipedia pages. Built with Streamlit and Google's Generative AI (Gemini).

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **📄 PDF Processing**: Upload and query multiple PDF documents
- **📝 Notion Integration**: Import and search through Notion exports (ZIP files)
- **🌐 Wiki Support**: Add Wikipedia or other wiki pages directly via URL
- **🤖 AI-Powered Q&A**: Get intelligent answers using Google's Gemini 2.5 Flash model
- **💬 Chat Interface**: Interactive chat interface with conversation history
- **🔍 Source Citations**: View the exact sources used to generate each answer
- **🎨 Modern UI**: Beautiful gradient design with smooth animations
- **⚡ Fast Vector Search**: Custom vector store implementation using cosine similarity

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google API Key (for Gemini AI)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd talk-to-syllabus
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## 📦 Dependencies

```
streamlit
PyPDF2
langchain
langchain-google-genai
python-dotenv
numpy
requests
beautifulsoup4
markdown
```

Install all dependencies:
```bash
pip install streamlit PyPDF2 langchain langchain-google-genai python-dotenv numpy requests beautifulsoup4 markdown
```

## 🔑 Getting Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key
5. Add it to your `.env` file

## 💡 How to Use

### 1. Upload Content

**PDF Files:**
- Click on "📄 PDF Files" in the sidebar
- Drag and drop or browse to select PDF files
- Multiple PDFs can be uploaded at once

**Notion Exports:**
- Export your Notion workspace as Markdown & CSV (ZIP format)
- Click on "📝 Notion Exports" in the sidebar
- Upload the ZIP file

**Wiki Pages:**
- Click on "🌐 Wiki Pages" in the sidebar
- Paste a Wikipedia or wiki URL
- Click "➕ Add"

### 2. Ask Questions

Once content is uploaded:
- Type your question in the chat input at the bottom
- Press Enter or click Send
- The AI will analyze your content and provide an answer
- Click "🔍 Show Sources" to see which parts of your documents were used

### 3. Manage Your Session

- **🔄 Reset Chat**: Clear conversation history while keeping uploaded content
- **🗑 Clear All**: Remove all uploaded content and reset the entire session

## 🏗️ Architecture

### Components

1. **Content Processors**
   - `process_pdf_file()`: Extracts and chunks PDF content
   - `process_notion_export()`: Processes Notion markdown files
   - `process_wiki_url()`: Scrapes and processes web content

2. **Vector Store**
   - `SimpleVectorStore`: Custom implementation using cosine similarity
   - No FAISS dependency (works on Python 3.13+)
   - Efficient document retrieval

3. **AI Model**
   - Google Gemini 2.5 Flash for embeddings and chat
   - Context-aware responses
   - Source attribution

### Data Flow

```
Upload Content → Process & Chunk → Create Embeddings → Store in Vector DB
                                                              ↓
User Question → Embed Query → Search Similar Chunks → Generate Answer
```

## 🎨 Customization

### Modify Chunk Size

In the text splitter configuration:
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust this
    chunk_overlap=100     # Adjust this
)
```

### Change AI Model

In the `get_answer_simple()` function:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Change model here
    temperature=0.1,            # Adjust creativity
    max_output_tokens=1024,     # Adjust response length
)
```

### Customize Number of Sources

In similarity search:
```python
matching_chunks = vector_store.similarity_search(user_query, k=2)  # Change k value
```

## 🛠️ Troubleshooting

### API Key Issues
- Ensure your `.env` file is in the project root directory
- Verify the API key is valid and has the necessary permissions
- Check that `python-dotenv` is installed

### PDF Processing Errors
- Ensure PDFs are not password-protected
- Try uploading one PDF at a time
- Check that PDFs contain extractable text (not scanned images)

### Memory Issues
- Reduce `chunk_size` in text splitter
- Process fewer documents at once
- Clear unused content regularly

### Import Errors
```bash
# If you get import errors, reinstall dependencies:
pip install --upgrade streamlit langchain langchain-google-genai
```

## 📝 Project Structure

```
talk-to-syllabus/
│
├── app.py                 # Main application file
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore file
├── README.md             # This file
├── requirements.txt      # Python dependencies
│
└── (generated at runtime)
    ├── chat_history      # Session conversation data
    └── vector_store      # Embedded document vectors
```

## Screenshots

<img width="1861" height="876" alt="Screenshot 2026-02-16 191529" src="https://github.com/user-attachments/assets/d6331ee3-d221-453c-a57d-ed71417fac47" />
<br>
<img width="1320" height="658" alt="Screenshot 2026-02-16 191718" src="https://github.com/user-attachments/assets/45b06c67-2340-49f8-813f-623d1c340e44" />

</br>
<img width="1341" height="622" alt="Screenshot 2026-02-16 191745" src="https://github.com/user-attachments/assets/e3dfd56c-9e0e-48e8-80a2-66b68d466132" />
</br>
<img width="1350" height="449" alt="Screenshot 2026-02-16 191544" src="https://github.com/user-attachments/assets/aa54674b-a885-439f-a483-10ae0cb20897" />
</br>
<img width="1341" height="622" alt="Screenshot 2026-02-16 191745" src="https://github.com/user-attachments/assets/06249a85-b12d-4529-a7aa-0d71d45a12ad" />




## 🔒 Security Notes

- **Never commit your `.env` file** - it contains your API key
- Add `.env` to `.gitignore`
- Use environment variables for sensitive data
- Keep your Google API key secure

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Generative AI](https://ai.google.dev/)
- Uses [LangChain](https://python.langchain.com/) for document processing
- Inspired by the need for intelligent document Q&A systems

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for students and knowledge workers**
