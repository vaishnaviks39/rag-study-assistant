# %pip install streamlit openai langchain langchain-community langchain-openai faiss-cpu pypdf tiktoken

from pypdf import PdfReader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from tiktoken import get_encoding

# Load PDF
def load_data(uploaded_file):
  reader = PdfReader(uploaded_file)
  text = ""
  for page in reader.pages:
    text += page.extract_text()
  return text

# Split Text
def split_text(text, chunk_size, chunk_overlap):
  splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

  chunks = splitter.split_text(text)

  #convert chunks into documents
  docs = []
  for chunk in chunks:
    doc = Document(page_content=chunk)
    docs.append(doc)

  return docs

# Initialize OpenAI Client
def initialize_llm(openai_api_key):
  return OpenAI(api_key=openai_api_key)

# Prompt for question generation
QUESTION_PROMPT = """
You are an expert exam tutor.

Based on the study material below, generate high-quality exam preparation questions
that cover all important concepts.

TEXT:
------------
{text}
------------

Instructions:
- Do not add explanations or answers
- Do not introduce outside knowledge
- Return one clear question per line

QUESTIONS:
"""

def generate_questions(client, documents):
  # Combine all document text
    full_text = "\n\n".join(doc.page_content for doc in documents)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=QUESTION_PROMPT.format(text=full_text)
    )

    return response.output_text

def create_vector_store(openai_api_key, documents):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vector_store

def answer_question(client, vector_store, question):
    # Step 1: retrieve relevant chunks
    docs = vector_store.similarity_search(question, k=4)

    # Step 2: build context from retrieved chunks
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 3: grounded prompt
    prompt = f"""
You are a helpful study assistant.
Answer the question using ONLY the context below.

CONTEXT:
------------
{context}
------------

QUESTION:
{question}

ANSWER:
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text