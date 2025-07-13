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

from PyPDF2 import PdfReader  # For in-memory PDF parsing

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Function to format output nicely
def format_answer(raw_text):
    formatted = raw_text
    formatted = re.sub(r"(?m)^\s*(\d+)\)", r"\1.", formatted)
    formatted = formatted.replace("* ", "- ")
    formatted = re.sub(r"(?m)^(.*?:)\s*$", r"### \1", formatted)
    formatted = formatted.replace("\n\n", "\n\n---\n\n")
    return formatted

# Streamlit setup
st.set_page_config(page_title="Doc Q&A with Groq + ChromaDB")
st.title("Ask Questions from PDF or TXT using Groq + ChromaDB")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.read()
    file_extension = file_name.split(".")[-1]

    st.success(f"{file_name} uploaded successfully!")

    try:
        # Load and split content
        if file_extension == "pdf":
            # Read PDF content directly from memory
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif file_extension == "txt":
            text = file_bytes.decode("utf-8")
        else:
            st.error("Unsupported file type")
            st.stop()

        if not text.strip():
            st.warning("No extractable text found in the file.")
            st.stop()

        # Split into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.create_documents([text])

        # Generate embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

        # Initialize LLM and QA chain
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        # User query
        query = st.text_input("Ask a question about your document:")

        if query:
            with st.spinner("Generating answer..."):
                raw_answer_dict = qa_chain.invoke(query)
                raw_answer = raw_answer_dict.get('result', '')
                formatted_answer = format_answer(raw_answer)

                st.markdown("### Answer")
                st.markdown(formatted_answer)

                # Download answer as text file
                st.download_button(
                    label="Download Answer as TXT",
                    data=io.BytesIO(formatted_answer.encode("utf-8")),
                    file_name="answer.txt",
                    mime="text/plain"
                )

                # Download original file again (optional)
                st.download_button(
                    label="Download Uploaded File",
                    data=file_bytes,
                    file_name=file_name,
                    mime=uploaded_file.type
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
