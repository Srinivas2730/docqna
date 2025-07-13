import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="Doc Q&A with Groq + Chroma")
st.title("📄 Ask Questions from PDF using Groq + ChromaDB")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading and processing document..."):
        # Save file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF and split into pages
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()

        # Split into smaller text chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # Use Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create Chroma vector store
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

        # Setup Groq LLM
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

        # Retrieval-based QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        st.success("✅ Document processed! You can now ask questions.")

        query = st.text_input("💬 Ask a question about your document")

        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.invoke(query)
                st.write("🧠 Answer:")
                st.success(answer)


