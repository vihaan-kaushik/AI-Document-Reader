# 🧠 AI Document Reader

An advanced **AI-powered document understanding system** that reads, indexes, and answers questions from your documents using **large language models (LLMs)** like **Mistral-7B** and the **LangChain** framework.

---

## 🚀 Features

- 📄 **Multi-format Support** – Reads and processes `.pdf`, `.docx`, `.pptx`, and `.txt` files.
- 🧠 **LLM Integration** – Uses `Mistral-7B` for natural language understanding and response generation.
- ⚡ **Vector Search** – Leverages FAISS/ChromaDB for efficient semantic retrieval.
- 🔍 **Context-aware Q&A** – Ask any question related to your documents.
- 💾 **Optimized Loading** – Automatically adapts between 4-bit quantization (GPU) or CPU fallback.
- 🧩 **LangChain Framework** – Modular and extendable pipeline for document ingestion and querying.

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Language Model | [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
| Framework | [LangChain](https://www.langchain.com/) |
| Embeddings | `sentence-transformers` |
| Databases | `FAISS` / `ChromaDB` |
| Document Parsing | `pypdf`, `python-docx`, `python-pptx` |
| Backend Language | Python 3.10+ |
| Libraries | Transformers, Accelerate, Torch, LangChain-Community |

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/AI_Document_Reader.git
cd AI_Document_Reader
pip install -r requirements.txt
```

Or install directly in a Jupyter/Colab notebook:

```python
!pip install -U transformers accelerate langchain langchain-community sentence-transformers faiss-cpu chromadb pypdf python-docx python-pptx
```

---

## 🧭 Usage

1. **Load the model and tokenizer**
   ```python
   tokenizer, model = load_mistral_safely("mistralai/Mistral-7B-v0.1")
   ```

2. **Ingest your document**
   ```python
   from langchain_community.document_loaders import PyPDFLoader
   docs = PyPDFLoader("sample.pdf").load()
   ```

3. **Build embeddings and store**
   ```python
   from langchain.vectorstores import FAISS
   vector_store = FAISS.from_documents(docs, embedding_function)
   ```

4. **Ask questions**
   ```python
   query = "Summarize key insights from the document"
   results = vector_store.similarity_search(query)
   ```

---

## 🧠 How It Works

1. **Document Loading** – Reads supported files and extracts text.
2. **Chunking** – Splits text into manageable segments.
3. **Embedding** – Converts text into numerical vectors using `sentence-transformers`.
4. **Vector Search** – Uses FAISS/ChromaDB for similarity matching.
5. **LLM Reasoning** – Uses Mistral-7B to generate contextual, human-like answers.

---

## 🧰 Example Use Cases

- Intelligent document Q&A system
- Automated meeting summaries
- Legal or research paper analysis
- Knowledge base indexing
- Educational content understanding

---

## 📦 Requirements

- Python ≥ 3.10
- At least 8GB RAM (for CPU)
- GPU recommended for faster performance (with 4-bit quantization)

---

## 🧑‍💻 Author

**Vihaan Kaushik**  
B.Tech IT, ABES Engineering College  
📧 workvihu@gmail.com  

---

## 🪪 License

This project is licensed under the **MIT License** – feel free to use and modify with attribution.

---

## 🌟 Future Improvements

- Add web UI using Streamlit or Gradio  
- Multi-document summarization  
- Fine-tuned model integration  
- Local database caching for offline retrieval  
