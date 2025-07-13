                                🌐 Streamlit Project
                        📄 Doc Q&A App using Groq + ChromaDB

📌 Project Overview

Welcome to my Document Q&A Web App! This app lets you upload PDF or TXT files and ask questions about their content. It uses Groq’s powerful LLaMA3 model combined with ChromaDB vector store for lightning-fast and accurate answers — all powered by Streamlit for an easy, interactive experience.

🚀 How I Built & Ran the App (Step-by-Step):

Here’s the exact process I followed to bring this project to life 👇

1️⃣ Create a fresh project folder for the Doc Q&A app.

2️⃣ Inside that folder, create a file named docqna_chroma.py

🧠 This file contains the Streamlit app logic that handles document upload, embedding, and question answering.

3️⃣ Add a .env file
🔐 Use this file to securely store your API key:

GROQ_API_KEY=your_actual_groq_api_key_here

4️⃣ Create a requirements.txt file including:

streamlit

python-dotenv

langchain-groq

langchain-community

langchain-huggingface

chromadb

5️⃣ Open your terminal inside your project directory.

6️⃣ Install all the required packages:

pip install -r requirements.txt

7️⃣ Run the application using Streamlit:

streamlit run docqna_chroma.py

8️⃣ Visit the app in your browser:

🌍 http://localhost:8501 (or the port Streamlit shows)

🔁 GitHub Upload Steps

(How I published my project to GitHub 💻)

1️⃣ Created a new repository on GitHub.

2️⃣ Opened terminal inside my local project folder.

3️⃣ Initialized Git locally: git init

4️⃣ Added all project files: git add .

5️⃣ Committed changes with a message:

git commit -m "Add Doc Q&A app with Groq + ChromaDB"

6️⃣ Linked the folder to GitHub:

git remote add origin https://github.com/your-username/your-repo-name.git

7️⃣ Pushed everything to GitHub:

git push -u origin main

📝 Replace your-username and your-repo-name with your actual GitHub info.

📁 Project Folder Structure

📦 doc-qna-chroma-app

┣ 📄 docqna_chroma.py → Main Streamlit app with Doc Q&A logic

┣ 📄 .env → Securely stores Groq API key

┣ 📄 requirements.txt → Python dependencies list

┣ 📄 README.md → This setup guide and documentation

💡 What the App Can Do

✔ Upload PDF or TXT documents

✔ Use Groq’s LLaMA3 model for fast, contextual answers

✔ Utilize ChromaDB vectorstore for efficient retrieval

✔ Display answers in a clean, formatted way

✔ Download the original uploaded file or the answer as a TXT file

✔ Interactive and minimal UI built with Streamlit

✨ Tech Stack Used

Streamlit — frontend UI and interactivity

Groq API with LLaMA3 — language model for Q&A

ChromaDB — vector database for document embeddings

Python-dotenv — secure API key management

👩‍💻 Created By

Ushmitha Annapaneni

Feel free to ⭐ star or fork the project if you find it interesting!

📄 License

MIT License – Free to use, modify, and share

