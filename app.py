import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Embed chunks and store in FAISS
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Load LangChain conversational QA chain
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question using only the context below. If the answer is not in the context, say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user questions and display conversation
def user_input(user_question, api_key, pdf_docs, conversation_history):
    if not api_key or not pdf_docs:
        st.warning("Please upload PDF(s) and enter your API key.")
        return

    text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(text)
    get_vector_store(chunks, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]

    pdf_names = ", ".join([pdf.name for pdf in pdf_docs])
    conversation_history.append((user_question, answer, "Google AI", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pdf_names))

    # Chat UI rendering
    for question, reply, _, timestamp, _ in reversed(conversation_history):
        st.markdown(f"""
        <div style="padding:1em;margin:1em 0;background:#2b313e;border-radius:8px;color:#fff;">
        <b>User:</b> {question}<br><br><b>Answer:</b> {reply}<br><span style="font-size:0.8em;">{timestamp}</span></div>
        """, unsafe_allow_html=True)

    # Download button
    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
    st.title("📄 Chat with your PDF files")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Google API Key", type="password")
        st.markdown("[Get your API key](https://ai.google.dev/)")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if uploaded_files:
                st.success("PDFs processed successfully.")
            else:
                st.warning("Upload PDFs first.")

        if st.button("Reset"):
            st.session_state.conversation_history = []

    user_question = st.text_input("Ask a question about your PDFs")

    if user_question and uploaded_files and api_key:
        user_input(user_question, api_key, uploaded_files, st.session_state.conversation_history)
        st.session_state.user_question = ""

if __name__ == "__main__":
    main()
