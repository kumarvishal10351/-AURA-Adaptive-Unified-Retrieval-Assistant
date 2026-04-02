# 🤖 RAG Assistant (AI Research Copilot)

A **production-grade Retrieval-Augmented Generation (RAG) assistant** built using **Mistral, LangChain, ChromaDB, and Streamlit**.
This app allows users to **chat with their documents intelligently**, combining LLM reasoning with vector search.

---

## 🚀 Features

- 💬 Chat with your documents (PDF, text, etc.)
- 🧠 Powered by **Mistral LLM**
- 🔍 Semantic search using **ChromaDB**
- ⚡ Fast retrieval with embeddings
- 🎨 Modern Streamlit UI (ChatGPT-style)
- 🧩 Modular architecture (ingestion, retrieval, LLM, utils)
- 📊 Confidence scoring for responses
- 🔄 Fallback handling for robustness

---

## 🧠 How It Works

1. Documents are loaded and split into chunks
2. Embeddings are generated and stored in ChromaDB
3. User query is converted into embeddings
4. Relevant chunks are retrieved
5. LLM generates a context-aware response

---

## 🛠️ Tech Stack

- **LLM:** Mistral (via API)
- **Framework:** LangChain
- **Vector DB:** ChromaDB
- **Frontend:** Streamlit
- **Language:** Python

---

## 📂 Project Structure

```bash
rag-assistant/
│
├── app/
│   ├── main.py                # Streamlit app entry point
│   ├── config/
│   │   └── settings.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   └── embedder.py
│   ├── llm/
│   │   ├── mistral_client.py
│   │   └── fallback.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── utils/
│   │   ├── helpers.py
│   │   └── confidence.py
│
├── data_docs/                 # Input documents
├── chroma_db/                # Vector database (ignored in git)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/rag-assistant.git
cd rag-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app/main.py
```

---

## 📸 Demo (Optional)

_Add screenshots or GIFs of your UI here_

---

## 🧪 Example Use Cases

- Research assistant for PDFs
- AI-powered document Q&A
- Knowledge base chatbot
- Personal AI study assistant

---

## 🚧 Future Improvements

- 🔊 Voice input/output
- 📂 Drag-and-drop file upload
- 🧠 Long-term memory
- 🌐 Multi-user support
- 📊 Analytics dashboard

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a PR.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Vishal Kumar Kashyap**
Aspiring AI Engineer | Building RAG & AI systems

---

## ⭐ Support

If you like this project:

- ⭐ Star this repo
- 🍴 Fork it
- 📢 Share it

---
