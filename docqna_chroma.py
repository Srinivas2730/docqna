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

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def format_answer(raw_text):
    """
    Formats raw LLM output text into a cleaner Markdown string for Streamlit display.
    """
    formatted = raw_text
    # Convert numbered lists from e.g. "1)" to "1."
    formatted = re.sub(r"(?m)^\s*(\d+)\)", r"\1.", formatted)
    # Replace '* ' with '- ' for markdown bullet lists
    formatted = formatted.replace("* ", "- ")
    # Add Markdown heading style for lines ending with colon
    formatted = re.sub(r"(?m)^(.*?:)\s*$", r"### \1", formatted)
    # Add horizontal rules between paragraphs
    formatted = formatted.replace("\n\n", "\n\n---\n\n")
    return formatted

st.set_page_config(page_title="Doc Q&A with Groq + ChromaDB")
st.title("📄 Ask Questions from PDF or TXT using Groq + ChromaDB")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

    # Save uploaded file temporarily
    temp_filename = "temp_file." + uploaded_file.name.split(".")[-1]
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)

    # Load documents based on file type
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(temp_filename)
        pages = loader.load_and_split()
    elif uploaded_file.type == "text/plain":
        loader = TextLoader(temp_filename, encoding="utf8")
        pages = loader.load_and_split()
    else:
        st.error("Unsupported file type")
        st.stop()

    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    # Initialize Groq LLaMA3 LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    query = st.text_input("💬 Ask a question about your document")

    if query:
        with st.spinner("Generating answer..."):
            raw_answer_dict = qa_chain.invoke(query)
            raw_answer = raw_answer_dict.get('result', '')  # Extract text safely

            formatted_answer = format_answer(raw_answer)

            st.write("🧠 Answer:")
            st.markdown(formatted_answer)

            # Download uploaded file button
            st.download_button(
                label="📥 Download Uploaded File",
                data=file_bytes,
                file_name=uploaded_file.name,
                mime=uploaded_file.type
            )

            # Download answer as TXT button
            answer_bytes = formatted_answer.encode("utf-8")
            answer_buffer = io.BytesIO(answer_bytes)
            st.download_button(
                label="📥 Download Answer as TXT",
                data=answer_buffer,
                file_name="answer.txt",
                mime="text/plain"
            )



