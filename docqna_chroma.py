import streamlit as st
from dotenv import load_dotenv
import os
import re
import io

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def format_answer(raw_text):
    formatted = raw_text
    formatted = re.sub(r"(?m)^\s*(\d+)\)", r"\1.", formatted)
    formatted = formatted.replace("* ", "- ")
    formatted = re.sub(r"(?m)^(.*?:)\s*$", r"### \1", formatted)
    formatted = formatted.replace("\n\n", "\n\n---\n\n")
    return formatted

st.set_page_config(page_title="Doc Q&A with Groq + ChromaDB")
st.title("📘 Ask Questions from PDF or TXT using Groq + ChromaDB")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    temp_filename = "temp_file." + uploaded_file.name.split(".")[-1]

    # Save temp file
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)

    st.success(f"{uploaded_file.name} uploaded successfully!")

    try:
        # Load file
        loader = PyPDFLoader(temp_filename) if uploaded_file.type == "application/pdf" else TextLoader(temp_filename, encoding="utf8")
        pages = loader.load_and_split()

        # Split into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # Embeddings & vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

        # LLM + Chain
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        query = st.text_input("🔍 Ask a question about your document")

        if query:
            with st.spinner("Generating answer..."):
                raw_answer_dict = qa_chain.invoke(query)
                raw_answer = raw_answer_dict.get('result', '')
                formatted_answer = format_answer(raw_answer)

                st.write("**Answer:**")
                st.markdown(formatted_answer)

                # Download original file
                st.download_button(
                    label="Download Uploaded File",
                    data=file_bytes,
                    file_name=uploaded_file.name,
                    mime=uploaded_file.type
                )

                # Download answer
                answer_bytes = formatted_answer.encode("utf-8")
                st.download_button(
                    label="Download Answer as TXT",
                    data=io.BytesIO(answer_bytes),
                    file_name="answer.txt",
                    mime="text/plain"
                )
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
