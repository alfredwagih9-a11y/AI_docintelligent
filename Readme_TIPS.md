# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Alfred wagih             |
| Project Name     | AI Document Intelligence System (RAG + Output Parser) |
| GitHub Username  | Alfred Wagih     |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

**AI Document Intelligence System** lets you upload documents (contracts, reports, lab sheets, any text-based PDF) and ask questions about them in plain language.

It uses **Retrieval-Augmented Generation (RAG)**: the PDFs are split into chunks, converted into embeddings and stored in a vector database. When you ask a question, the system retrieves only the most relevant chunks and gives them to a local LLM, so answers are based on your documents instead of the model's memory.

Every answer is returned through a **structured output parser**, so the result always has the same clean format:

**Document → Summary → Key Points → Risks → References (file and page)**

The whole system is **100% free and runs locally**: no API keys and no paid services.

---

# ✨ Features

* **PDF loading**: upload one or many PDFs; text is extracted page by page.
* **Question Answering**: ask about all documents or a single document, with the retrieved context shown for transparency.
* **Summarization**: get a summary, key points and risks for a chosen document.
* **Document Comparison**: compare two documents and get similarities, differences and risks.
* **Structured output**: answers are validated with Pydantic, so the format is consistent.
* **References**: every answer points to the source file and page number.
* **Streamlit GUI**: simple web interface with a sidebar for settings and tabs for each task.
* **Free and private**: documents never leave your computer.

---

# 🛠️ Technologies Used

| Component | Tool |
| --- | --- |
| Language | Python 3.10+ |
| GUI | Streamlit |
| LLM (free, local) | Ollama (`llama3.2`, `qwen2.5`, `mistral`, ...) |
| Framework | LangChain (prompts, chains, retriever) |
| PDF loader | PyPDF (`PyPDFLoader`) |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | FAISS (`faiss-cpu`) |
| Output parser | Pydantic + LangChain `PydanticOutputParser` |

---

# ⚙️ Installation

1. **Install Python 3.10 or newer** from [python.org](https://www.python.org/) (tick "Add Python to PATH" on Windows).

2. **Install Ollama** from [ollama.com/download](https://ollama.com/download) and download a model:

   ```bash
   ollama pull llama3.2
   ```

3. **Clone or unzip the project**, then open a terminal in the project folder.

4. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac / Linux
   ```

5. **Install the libraries**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Run the app**:

   ```bash
   streamlit run app.py
   ```

   The browser opens at `http://localhost:8501`.

---

# 🚀 Usage

1. Make sure **Ollama is running** and the model is downloaded (`ollama list`).
2. In the **sidebar**, upload one or more PDF files.
3. Click **Process documents** and wait for the indexing to finish.
4. Choose a tab:
   * **💬 Question Answering**: type a question, for example *"What are the termination conditions?"*.
   * **📝 Summarization**: select a document and click **Summarize**.
   * **🔀 Compare Documents**: select two documents, optionally enter a focus, and click **Compare**.
5. Read the structured result: Document, Summary, Key Points, Risks and References.

**How it works internally**

```
PDF upload → PyPDFLoader → text splitter → embeddings → FAISS vector DB
          → retriever → prompt → Ollama LLM → PydanticOutputParser → Streamlit UI
```

**Tips**

* If the model name is different, type the exact name from `ollama list` into the sidebar.
* For better and more reliable answers, use a larger model such as `qwen2.5:7b`.
* Scanned PDFs (images only) are not supported without OCR.

---


# 📈 Results

* Built a complete working RAG pipeline that runs fully offline with free tools.
* Answers are grounded in the uploaded documents and include file and page references.
* The output parser produces a consistent structure (Document, Summary, Key Points, Risks, References) for all three tasks.
* Added safeguards for small local models: optional fields with defaults, a regex fallback for messy JSON, and automatic references taken from the retrieved pages.
* Prompts were tuned so the model lists only risks that the document actually mentions, which reduced made-up risks.

---

# 🔮 Future Improvements

* Add OCR support for scanned PDFs and images.
* Support more file types (DOCX, TXT, PPTX).
* Add chat history and follow-up questions.
* Export results to PDF or Word.
* Add a map-reduce summarization chain for very long documents.
* Save the vector database to disk so documents don't need re-processing.
* Add multi-language support (Arabic and English).

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**, and it will be showcased on the official [Tips Hindawi](https://www.tipshindawi.com/) website.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
