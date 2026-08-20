# 🤖 ContextIQ – AI-powered document intelligence platform using Flask, RAG, FAISS, Docker and secure authentication.

An AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDF documents and ask questions grounded in their content.

The application uses **Sentence Transformers** for embeddings, **FAISS** for vector similarity search, and the **Groq API** with **OpenAI GPT-OSS 20B** to generate fast, context-aware answers through a secure Flask web application.

The application is fully **Dockerized** with persistent storage for uploaded documents, FAISS vector indexes, and SQLite-based user and chat data.

------------------------------------------------------------------------

## ✨ Features

-   🔐 JWT-based user authentication
-   📄 Upload PDF documents
-   📚 Automatic PDF text extraction
-   ✂️ Recursive text chunking
-   🧠 SentenceTransformer embeddings
-   ⚡ FAISS vector database for semantic search
-   🤖 Groq API with OpenAI GPT-OSS 20B
-   💬 Retrieval-Augmented Generation (RAG)
-   🌊 Streaming AI responses
-   🛡️ Flask-Limiter rate limiting
-   🗄️ SQLite database
-   💾 Persistent document and vector-store storage
-   🐳 Dockerized application
-   🔑 Environment-based configuration using .env
-   🎯 Answers based only on uploaded document context
-   🌙 Modern Dark Theme


------------------------------------------------------------------------

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Backend | Flask, Python |
| Database | SQLite, SQLAlchemy |
| Authentication | Flask-JWT-Extended |
| AI | Groq API, OpenAI GPT-OSS 20B |
| RAG | LangChain |
| Embeddings | Sentence Transformers |
| Vector Store | FAISS |
| PDF Processing | pypdf |
| Security | Flask-Limiter |
| Containerization | Docker |
| Frontend | HTML, CSS, JavaScript |

---------------------------------------------------------------------------

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
         Groq API (GPT-OSS 20B)
                 │
                 ▼
            Generated Answer(Stream Response)
```
```
Docker Architecture
                    Docker Container
                  ┌───────────────────┐
                  │     ContextIQ     │
                  │                   │
                  │ Flask Application │
                  │ Python Dependencies│
                  │ RAG + FAISS       |
                  |  ChatGroq         |
                  └─────────┬─────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
          user.db        uploads/      vector_store/
             │              │              │
             └──────────────┴──────────────┘
                  Persistent Bind Mounts
                            │
                            ▼
                      Host/ Volume
                            |
                            | HTTPS API
                            ▼
                       Groq API(GPT-OSS 20B)
```

------------------------------------------------------------------------

## 📂 Project Structure

``` text
AI_Chatbot/
│── app.py
│── database.py
|── .dockerignore
|── Dockerfile
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

Configure Groq

Get your API key from the Groq Console.

Create a .env file in the project root:

```
GROQ_API_KEY=your_groq_api_key
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
🐳 Docker Setup

ContextIQ can also be run inside a Docker container.

Build the Docker Image

From the project root:

```
docker build -t contextiq .
```
Run the Container

The application uses runtime environment variables and persistent bind mounts for the SQLite database, uploaded documents, and FAISS vector store.

For PowerShell:

```
docker run --env-file .env `
  --mount type=bind,source="${PWD}\uploads",target=/app/uploads `
  --mount type=bind,source="${PWD}\vector_store",target=/app/vector_store `
  --mount type=bind,source="${PWD}\user.db",target=/app/user.db `
  -p 5000:5000 contextiq
```
Open:

```
http://localhost:5000
```
The GROQ_API_KEY is supplied to the container at runtime through the .env file. 
The API key is not included in the Docker image.

------------------------------------------------------------------------

## 📸 Screenshots

### home

![Home](screenshot/PolishedHome.png)

### Signup

![Signup](screenshot/polishedsignin.png)

### Login

![Login](screenshot/polishedlogin.png)

### Chat Interface

![Chat](screenshot/polishedchatbot.png)

------------------------------------------------------------------------

## 💡 How It Works
- Register or log in.
- Upload a PDF document.
- The application extracts and chunks the text.
- Embeddings are generated using Sentence Transformers.
- Chunks are indexed in FAISS.
- Ask questions about the uploaded document.
- Relevant chunks are retrieved using semantic similarity.
- The retrieved context and question are sent to the Groq API.
- GPT-OSS 20B generates a grounded response.
- The response is streamed back to the user.
- Chat history is persisted in SQLite.

------------------------------------------------------------------------
## 🔮 Future Improvements
- 🚀 Production deployment
- 🌐 Cloud deployment
- 📊 Monitoring and logging
- 🔐 Role-based access control
- 🐳 Docker Compose configuration
- ⚙️ Production WSGI server
------------------------------------------------------------------------

## 👨‍💻 Author

**Rutendra Mahato**

If you found this project useful, consider giving it a ⭐ on GitHub.
