import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load env vars
load_dotenv()

# Debug prints (temporary)
print("LANGSMITH_API_KEY:", os.getenv("LANGSMITH_API_KEY"))
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{question}")
    ]
)

# Streamlit UI
st.title("Ollama + LangSmith (phi)")
question = st.text_input("Ask something")

# Local LLM
llm = Ollama(
    model="phi",
    timeout=120
)

# Chain
chain = prompt | llm | StrOutputParser()

# Run
if question:
    st.write(chain.invoke({"question": question}))
