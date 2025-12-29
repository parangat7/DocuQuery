# 📄 DocuQuery: Enterprise RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents.

## 🚀 Architecture Highlights
This project demonstrates a **Hybrid AI Architecture** optimized for cost and privacy:
* **Embeddings:** Uses `all-MiniLM-L6-v2` (Local HuggingFace model). This runs locally on the CPU, ensuring document vectors are generated with **zero API cost** and higher privacy.
* **Vector Store:** FAISS (Facebook AI Similarity Search) for efficient local similarity search.
* **LLM:** Google Gemini 1.5 Flash (via API) for high-speed, low-latency answer generation.

## 🛠️ Tech Stack
* **Python 3.10+**
* **Streamlit** (Frontend)
* **LangChain** (Orchestration)
* **FAISS** (Vector Database)
* **Google Gemini API**

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/DocuQuery.git](https://github.com/yourusername/DocuQuery.git)

## 🌐 Live Demo

You can try out the live application here: [DocuQuery on Streamlit Cloud](https://docuquery930.streamlit.app)   
