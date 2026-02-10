# StudyMate AI: Smart Exam Prep Assistant

StudyMate AI is a Streamlit-based application that helps students prepare for exams by generating practice questions and answering selected questions directly from uploaded PDF study material. The system uses a Retrieval-Augmented Generation (RAG) pipeline to ensure responses are grounded in the uploaded document content.

This project is Dockerized and can be deployed on cloud platforms such as AWS and Google Cloud Platform (GCP).

---

## Features

- Upload PDF textbooks, lecture notes, or study material
- Automatically generate exam-style questions from the PDF content
- Select questions and generate answers grounded in the document
- Semantic search using FAISS vector database
- OpenAI embeddings for efficient document retrieval
- Modular architecture with separate RAG pipeline and UI components
- Dockerized setup for easy deployment on AWS/GCP

---

## Tech Stack

- Python
- Streamlit
- OpenAI API (Responses API)
- LangChain
- FAISS Vector Store
- OpenAI Embeddings
- pypdf (PDF text extraction)
- Docker

---

## Project Structure

```bash
studymate-app/
│
├── app.py                # Streamlit application UI
├── rag_pipeline.py       # Core RAG pipeline functions
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration for deployment
├── .dockerignore         # Files ignored during Docker build
└── README.md             # Project documentation
```

---

## Deployment

This project is Dockerized and can be deployed on cloud platforms.

### Deployment Options
- AWS (App Runner / ECS)
- Google Cloud Platform (Cloud Run)

Since the app runs inside a Docker container, deployment is portable across cloud providers.

---

## How It Works (RAG Pipeline Overview)

- The uploaded PDF is parsed using `pypdf`.
- The extracted text is split into chunks using LangChain token-based splitting.
- OpenAI embeddings are generated for each chunk.
- FAISS is used as a vector database for semantic similarity search.
- When a user asks a question, the most relevant chunks are retrieved.
- OpenAI generates an answer using only the retrieved context.

---

## Future Improvements

- Streaming responses for real-time answer generation
- Support for multiple file uploads
- Citation-based answers with referenced document sections
- User authentication for multi-user deployment
