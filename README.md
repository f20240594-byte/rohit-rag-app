# 📄 RAG Chatbot — Chat with Your Documents

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and Groq.  
Upload a PDF, DOCX, or TXT file and ask questions about it — the app finds the most relevant sections and answers using an LLM.

---

## What document did you use and why?

I used a **PDF of my college notes / a technical report** as the test document.  
I chose it because it's text-heavy and has specific facts that are easy to verify — making it a great way to test whether the chatbot is actually retrieving the right information rather than just guessing.

---

## How does your chunking work?

The document text is split into **overlapping fixed-size chunks** using a sliding window approach:

- **Chunk size:** 1000 characters per chunk
- **Overlap:** 100 characters shared between consecutive chunks

The overlap ensures that sentences or ideas that fall on a chunk boundary are not lost — both neighboring chunks will contain that context.

```
[ chunk 1: 0 → 1000 ]
          [ chunk 2: 900 → 1900 ]
                    [ chunk 3: 1800 → 2800 ]
```

This is handled in `utils/loader.py` → `split_into_chunks()`.

---

## Which embedding model did you use?

I used **`all-MiniLM-L6-v2`** from the `sentence-transformers` library.

| Property | Detail |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Library | `sentence-transformers` |
| Embedding size | 384 dimensions |
| Why this model? | Fast, lightweight, free, and works well for semantic search tasks |

Chunks are embedded once when the document is uploaded and stored in session state. At query time, the question is embedded and compared against all chunk embeddings using **cosine similarity** to find the most relevant chunks.

---

## How to run locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/rohit-rag-app.git
cd rohit-rag-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free key at [console.groq.com](https://console.groq.com)

**4. Run the app**
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Screenshot

<img width="1892" height="837" alt="demo png" src="https://github.com/user-attachments/assets/8f6ead62-ea08-4ef7-8edf-d66a78990233" />


---

## What would you improve with more time?

- **Better chunking** — use sentence-aware splitting (e.g. split on `.` or `\n`) instead of fixed character count, so chunks don't cut mid-sentence
- **Vector database** — replace in-memory numpy arrays with a proper vector store like FAISS or ChromaDB for faster search on large documents
- **Multi-document support** — let users upload and query across multiple files at once
- **Chat memory** — make the LLM aware of the full conversation history for follow-up questions
- **Source highlighting** — show the user exactly which part of the document the answer came from
- **Better UI** — add a progress bar during embedding, and display chunk previews in an expandable section
