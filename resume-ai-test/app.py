import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# ----------------------------
# CONFIG
# ----------------------------
PDF_PATH = "/Users/apple/Desktop/LangChain/resume-ai-test/AryanSrivastava_Resume_December2025.pdf"
MODEL_NAME = "mistral"

st.set_page_config(page_title="Resume Interview Question Generator")
st.title("🧠 AI Interview Question Generator (Resume-based)")

# ----------------------------
# LOAD PDF
# ----------------------------
@st.cache_resource
def load_and_process_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb


# ----------------------------
# MAIN LOGIC
# ----------------------------
if st.button("📄 Analyze Resume & Generate Questions"):

    with st.spinner("Processing resume..."):
        vectordb = load_and_process_pdf()

    # Retrieve most relevant resume chunks
    relevant_docs = vectordb.similarity_search(
        "skills projects experience technologies",
        k=5
    )

    resume_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # LLM
    llm = Ollama(model=MODEL_NAME)

    prompt = f"""
You are a senior technical interviewer.

Based ONLY on the resume content below, generate exactly 10 interview questions.
Follow these rules:
- Questions should be technical + project-based
- Ask WHY and HOW, not just WHAT
- Mix easy, medium, and hard questions
- Avoid generic textbook questions

Resume Content:
----------------
{resume_context}
----------------

Return the questions as a numbered list.
"""

    with st.spinner("Generating interview questions..."):
        response = llm.invoke(prompt)

    st.subheader("📋 Interview Questions")
    st.write(response)
