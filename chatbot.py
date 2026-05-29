import requests
from PIL import Image
import io
import base64
import streamlit as st
from groq import Groq
import PyPDF2
import docx
import os

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 RAG Chatbot — Chat with Your Documents")

# ─── HELPER: Extract text from files ───────────────────────
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file)
    else:
        return None

# ─── HELPER: Split text into chunks ────────────────────────
def split_into_chunks(text, chunk_size=1000, overlap=100):
    """
    Splits text into overlapping chunks.
    overlap means chunks share some text so context isn't lost at boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ─── HELPER: Find relevant chunks for a question ───────────
def find_relevant_chunks(question, chunks, top_k=5):
    """
    Simple keyword search to find chunks most relevant to the question.
    Scores each chunk by how many question words it contains.
    """
    question_words = set(question.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)  # common words
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_k]]

# ─── HELPER: Generate image using Hugging Face ─────────────
def generate_image(prompt, hf_api_key=None):
    """
    Uses Pollinations AI — completely free, no API key needed!
    """
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"
    
    response = requests.get(image_url, timeout=60)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        return image
    else:
        return None

with st.sidebar:
    st.header("🔑 Groq API Key")
    api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        placeholder="gsk_..."
    )
    st.caption("Get your free key at console.groq.com")

    st.divider()

    st.header("🎨 Hugging Face Key")
    hf_api_key = st.text_input(
        "Enter your HuggingFace API key",
        type="password",
        placeholder="hf_..."
    )
    st.caption("Get your free key at huggingface.co")

    st.divider()

    model = st.selectbox(
        "🧠 Choose AI Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
    )

    st.divider()

    st.header("📁 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:
        with st.spinner("Reading document..."):
            text = extract_text(uploaded_file)
            if text:
                st.session_state.doc_text = text
                st.session_state.doc_chunks = split_into_chunks(text)
                st.session_state.doc_name = uploaded_file.name
                st.success(f"✅ {uploaded_file.name} loaded!")
                st.caption(f"📊 {len(st.session_state.doc_chunks)} chunks created")
            else:
                st.error("Could not read this file.")

    st.divider()
    # ─── TABS: Chat and Image Generation ──────────────────────
chat_tab, image_tab = st.tabs(["💬 Chat", "🎨 Generate Image"])

with image_tab:
    st.subheader("🎨 AI Image Generator")
    st.caption("Describe anything and AI will draw it for you!")

    image_prompt = st.text_area(
        "Describe the image you want",
        placeholder="e.g. A futuristic city at sunset, a cat wearing a space suit, a mountain lake at dawn...",
        height=100
    )

    if st.button("🎨 Generate Image", type="primary"):
        if not hf_api_key:
            st.error("⚠️ Please enter your Hugging Face API key in the sidebar!")
        elif not image_prompt:
            st.warning("Please describe the image you want!")
        else:
            with st.spinner("🎨 Creating your image... this takes ~20 seconds!"):
                result = generate_image(image_prompt, hf_api_key)

                if result == "loading":
                    st.warning("⏳ Model is warming up! Wait 30 seconds and try again.")
                elif result is None:
                    st.error("❌ Something went wrong. Check your API key and try again.")
                else:
                    st.image(result, caption=image_prompt, use_column_width=True)
                    st.success("✅ Image generated!")

                    # Save image to bytes for download
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label="⬇️ Download Image",
                        data=buf,
                        file_name="generated_image.png",
                        mime="image/png"
                    )

with chat_tab:

    if st.button("🗑️ Clear Chat", key="clear_chat_sidebar"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with Streamlit + Groq + HuggingFace ⚡")

    # ─── FILE UPLOADER ─────────────────────────────────────
    st.header("📁 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:
        with st.spinner("Reading document..."):
            text = extract_text(uploaded_file)
            if text:
                st.session_state.doc_text = text
                st.session_state.doc_chunks = split_into_chunks(text)
                st.session_state.doc_name = uploaded_file.name
                st.success(f"✅ {uploaded_file.name} loaded!")
                st.caption(f"📊 {len(st.session_state.doc_chunks)} chunks created")
            else:
                st.error("Could not read this file.")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with Streamlit + Groq ⚡")

# ─── SESSION STATE ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ─── SHOW DOCUMENT STATUS ──────────────────────────────────
if st.session_state.doc_name:
    st.info(f"📄 Document loaded: **{st.session_state.doc_name}** — Ask me anything about it!")

    # Summarize button
    if st.button("📝 Summarize This Document"):
        if not api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar!")
        else:
            with st.spinner("Summarizing..."):
                # Use first 4000 chars for summary (to stay within limits)
                preview = st.session_state.doc_text[:4000]
                client = Groq(api_key=api_key)
                summary_response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": f"Please summarize this document clearly and concisely:\n\n{preview}"
                    }],
                    max_tokens=1024,
                )
                summary = summary_response.choices[0].message.content
                with st.chat_message("assistant"):
                    st.markdown("### 📝 Document Summary")
                    st.markdown(summary)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"### 📝 Document Summary\n{summary}"
                })
else:
    st.warning("👈 Upload a document from the sidebar to get started — or just chat without one!")

st.divider()

# ─── DISPLAY PAST MESSAGES ─────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── CHAT INPUT ────────────────────────────────────────────
user_input = st.chat_input("Ask anything — about your document or anything else!")

if user_input:

    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar!")
        st.stop()

    # 1. Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 2. Build prompt — with or without document context
    client = Groq(api_key=api_key)

    if st.session_state.doc_chunks:
        # RAG: find relevant chunks and inject into prompt
        relevant = find_relevant_chunks(user_input, st.session_state.doc_chunks)
        context = "\n\n---\n\n".join(relevant)

        system_prompt = f"""You are a helpful assistant. 
The user has uploaded a document. Use the context below to answer their question.
If the answer is not in the context, say so honestly and answer from your own knowledge.

DOCUMENT CONTEXT:
{context}
"""
    else:
        # No document — normal chatbot
        system_prompt = "You are a helpful assistant."

    # 3. Call Groq
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages
                ],
                max_tokens=1024,
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })