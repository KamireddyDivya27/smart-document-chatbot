import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Fallback for Streamlit Cloud secrets
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
st.set_page_config(page_title="Smart Document Chatbot", page_icon="📄")
st.title("📄 Smart Document Chatbot")
st.markdown("Upload any PDF and ask questions about it!")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def get_vector_store(chunks):
    embeddings = load_embeddings()
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")


def answer_question(question):
    embeddings = load_embeddings()
    db = FAISS.load_local(
        "faiss_index", embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt_template = """You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "Answer not found in the document."
Do not make up answers.
Context:
{context}
Question: {question}
Answer:"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke(question)


with st.sidebar:
    st.header("Upload Documents")
    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"]
    )
    if st.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("Done! Now ask your question.")
        else:
            st.warning("Please upload at least one PDF.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    try:
        with st.spinner("Thinking..."):
            answer = answer_question(question)
    except Exception as e:
        answer = f"Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)