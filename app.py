import streamlit as st
import os

from rag_pipeline import (
    load_data,
    split_text,
    initialize_llm,
    generate_questions,
    create_vector_store,
    answer_question
)

st.set_page_config(
    page_title="StudyMate AI",
    layout="centered"
)

st.title("📘 StudyMate AI")
st.write(
    "Upload your study material (PDF), generate exam questions, "
    "and get answers grounded strictly in your notes."
)

# Session State Initialization
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.questions_text = None
    st.session_state.questions_list = []
    st.session_state.vector_store = None


# User Inputs

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")


uploaded_file = st.file_uploader(
    "Upload your study material (PDF)",
    type=["pdf"]
)

# Check if both API key and file are provided
if openai_api_key and uploaded_file:
    # Initialize OpenAI client
    client = initialize_llm(openai_api_key)

    if not st.session_state.initialized:
        # Load PDF
        with st.spinner("Reading PDF..."):
            text = load_data(uploaded_file)

        # Split text into chunks
        with st.spinner("Preparing document chunks..."):
            question_docs = split_text(
                text=text,
                chunk_size=8000,
                chunk_overlap=200
            )

            answer_docs = split_text(
                text=text,
                chunk_size=500,
                chunk_overlap=200
            )

        # Generate questions only once
        with st.spinner("Generating exam questions..."):
            st.session_state.questions_text = generate_questions(
                client,
                question_docs
            )

            st.session_state.questions_list = [
                q for q in st.session_state.questions_text.split("\n")
                if q.strip()
            ]

            st.session_state.vector_store = create_vector_store(
                openai_api_key,
                answer_docs
            )

        st.session_state.initialized = True

    # Display Generated Questions
    st.subheader("📋 Generated Exam Questions")
    st.info(st.session_state.questions_text)

    # Select Questions
    selected_questions = st.multiselect(
        "Select questions you want answers for:",
        st.session_state.questions_list
    )

    # Generate Answers
    if st.button("Generate Answers"):
        if selected_questions:
            for question in selected_questions:
                with st.spinner("Generating answer..."):
                    answer = answer_question(
                        client,
                        st.session_state.vector_store,
                        question
                    )

                st.markdown(f"### ❓ {question}")
                #st.success(answer)
                st.markdown(answer)
        else:
            st.warning("Please select at least one question.")

else:
    st.info("Please upload a PDF to begin.")