# PDF Knowledge Assistant

A streamlit-based application that allows you to upload PDF documents, extract text using Google Gemini, create embeddings, store them in Pinecone, and ask questions through an AI chatbot interface.

## Features

- 📄 **PDF Upload**: Upload single or multiple PDF documents
- 🤖 **Gemini Text Extraction**: Uses Google Gemini API for accurate text extraction from PDFs
- 🔍 **Semantic Search**: Uses text-embedding-3-small for creating embeddings
- 💾 **Vector Storage**: Stores embeddings in Pinecone for efficient retrieval
- 💬 **AI Chatbot**: Ask questions about your documents with GPT-4o
- 🎨 **Beautiful UI**: Clean, modern interface with preserved chat bubble styling

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add:
- `OPENAI_API_KEY`: Your OpenAI API key
- `GEMINI_API_KEY`: Your Google Gemini API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Your Pinecone index name

### 3. Create Pinecone Index

Create a Pinecone index with:
- **Dimension**: 1536 (for text-embedding-3-small)
- **Metric**: cosine

### 4. Run the Application

```bash
streamlit run app.py
```

## Usage

1. **Upload PDFs**: Go to "Upload PDFs" and select your PDF documents
2. **Wait for Processing**: The app will extract text, create embeddings, and store in Pinecone
3. **Ask Questions**: Go to "Chat Assistant" and ask questions about your documents
4. **Manage Database**: Use "Database Management" to view stats or reset the database

## Architecture

- **app.py**: Main Streamlit application with UI
- **pdf_processor.py**: PDF text extraction, chunking, embedding, and Pinecone upload
- **chatbot_utils.py**: RAG pipeline for query processing and response generation

## Tech Stack

- **Streamlit**: Web interface
- **Google Gemini**: PDF text extraction
- **OpenAI**: Text embeddings (text-embedding-3-small) and chat (GPT-4o)
- **Pinecone**: Vector database for semantic search
