


  
# DocuMind-RAG-Based-Intelligent-Document-QA-System

# 🔍 RAG Pipeline API

> 🚀 Upload documents. Ask questions. Get cited answers — powered by OpenAI GPT.

An end-to-end **Retrieval-Augmented Generation** system that performs **document ingestion**, **vector search**, and **AI-powered Q&A** on PDFs, DOCX, TXT, and Markdown files.

This project combines **FastAPI + ChromaDB + OpenAI + sentence-transformers** to build a real-world **document question-answering pipeline** used in domains like knowledge management, research, and enterprise search.

</div>

---

## 🧠 What Makes This Project Special?

Unlike traditional search systems, this project:

✔ Retrieves **exact relevant chunks at semantic level**

✔ Combines **vector search + LLM reasoning**

✔ Provides **source-cited, explainable answers**

✔ Works on **real-world multi-format documents**

✔ Is designed as a **decision-support system**, not just a model

---

## 🎯 Primary Use Cases

- **Enterprise Knowledge Base** 🏢
- **Legal Document Search** ⚖️
- **Research Paper Analysis** 🎓
- **Banking & KYC Document Q&A** 🏦
- **Customer Support Automation** 📑

---

## ⚙️ How It Works

1. Upload document (PDF / DOCX / TXT / MD)
2. Extract text from file
3. Split into overlapping chunks (512 chars, 64 overlap)
4. Embed chunks using `all-MiniLM-L6-v2`
5. Store embeddings in ChromaDB
6. On query → embed question → retrieve top-K chunks
7. Send chunks + question to OpenAI GPT
8. Return answer with source citations and relevance scores

---

## 🚀 Key Features

- Async background document processing ⚡
- Semantic vector search via ChromaDB 🔍
- Source-cited answers with relevance scores 🎯
- Fully interactive Swagger UI 📊
- Multi-format document support 📄
- Modular, production-ready architecture 🛠️

---

## 🧰 Tech Stack

- **Framework:** FastAPI + Uvicorn
- **LLM:** OpenAI GPT-4o-mini
- **Vector Store:** ChromaDB (local persistent)
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2`
- **Database:** SQLite + SQLAlchemy
- **Doc Parsing:** pypdf + python-docx
- **Container:** Docker + Docker Compose

---

## 📊 Sample Output

### 🔹 Health Check & Query Response

```json
{
  "status": "healthy",
  "vector_store": { "total_chunks": 108, "collection": "rag_documents" },
  "total_documents": 4
}
```

```json
{
  "query": "What is UNEP?",
  "answer": "UNEP is the United Nations Environment Programme, the foremost global organization dedicated to environmental protection, founded on June 5, 1972...",
  "sources": [{ "filename": "EVS_Assignment.pdf", "relevance_score": 0.6534 }],
  "model": "gpt-4o-mini",
  "chunks_retrieved": 5
}
```

---

## ▶️ Run the Project

```bash
git clone https://github.com/your-username/rag-pipeline-api.git
cd rag-pipeline-api
cp .env.example .env        # add your OPENAI_API_KEY
docker compose up --build
```

Open **`http://localhost:8000/docs`** for the interactive Swagger UI.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health and stats |
| `POST` | `/api/v1/documents/upload` | Upload a document |
| `GET` | `/api/v1/documents/` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document status |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + embeddings |
| `POST` | `/api/v1/query/` | Ask a question |

---

<div align="center">

Built with  using **FastAPI** · **OpenAI** · **ChromaDB** · **sentence-transformers**

</div>
