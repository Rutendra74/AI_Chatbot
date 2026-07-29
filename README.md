# 🤖 AI-Powered RAG Chatbot with Flask, FAISS & Ollama

An AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that
allows users to upload PDF documents and ask questions grounded in their
content. The application uses **Sentence Transformers** for embeddings,
**FAISS** for vector similarity search, and **Ollama (Llama 3.2)** to
generate context-aware answers through a secure Flask web application.

------------------------------------------------------------------------

## ✨ Features

-   🔐 JWT-based user authentication
-   📄 Upload PDF documents
-   📚 Automatic PDF text extraction
-   ✂️ Recursive text chunking
-   🧠 SentenceTransformer embeddings
-   ⚡ FAISS vector database for semantic search
-   🤖 Ollama (Llama 3.2) integration
-   💬 Retrieval-Augmented Generation (RAG)
-   🛡️ Flask-Limiter rate limiting
-   🗄️ SQLite database
-   🎯 Answers based only on uploaded document context

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Category         Technologies
  ---------------- -----------------------
  Backend          Flask, Python
  Database         SQLite
  Authentication   Flask-JWT-Extended
  AI               Ollama, Llama 3.2
  Embeddings       Sentence Transformers
  Vector Store     FAISS
  PDF Processing   PyPDF2
  Security         Flask-Limiter
  Frontend         HTML, CSS

------------------------------------------------------------------------

## 🏗️ System Architecture

``` text
            Upload PDF
                 │
                 ▼
        Extract PDF Text
                 │
                 ▼
      Recursive Text Chunking
                 │
                 ▼
    SentenceTransformer Embeddings
                 │
                 ▼
          FAISS Vector Store
                 │
                 ▼
        User asks a Question
                 │
                 ▼
      Retrieve Relevant Chunks
                 │
                 ▼
         Ollama (Llama 3.2)
                 │
                 ▼
            Generated Answer
```

------------------------------------------------------------------------

## 📂 Project Structure

``` text
AI_Chatbot/
│── app.py
│── database.py
│── main.py
│── requirements.txt
│── templates/
│── static/
│── uploads/
│── vector_store/
└── README.md
```

------------------------------------------------------------------------

## 🚀 Installation

``` bash
git clone <your-repository-url>
cd AI_Chatbot

python -m venv chatbot
chatbot\Scripts\activate

pip install -r requirements.txt
```

Start Ollama and pull the model:

``` bash
ollama pull llama3.2
ollama serve
```

Run the application:

``` bash
python app.py
```

Open:

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

## 💡 How It Works

1.  Register or log in.
2.  Upload a PDF document.
3.  The application extracts and chunks the text.
4.  Embeddings are generated using Sentence Transformers.
5.  Chunks are indexed in FAISS.
6.  Ask questions about the uploaded document.
7.  Relevant chunks are retrieved and provided to Ollama for grounded
    responses.

------------------------------------------------------------------------
chatbot.png

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Multiple document support
-   Conversation memory
-   Streaming responses
-   Docker deployment
-   Cloud vector database
-   Role-based access control

------------------------------------------------------------------------

## 👨‍💻 Author

**Rutendra Mahato**

If you found this project useful, consider giving it a ⭐ on GitHub.
