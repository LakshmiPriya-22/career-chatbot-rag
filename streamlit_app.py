import os
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# Streamlit page settings
st.set_page_config(
    page_title="AI Internship Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Internship Chatbot")

st.write("Ask questions about the internship PDF")


# Load PDF
loader = PyPDFLoader("data/Gudz_FullStack_AI_Intern_JD.pdf")

documents = loader.load()


# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ChromaDB
vector_db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="chroma_db"
)


# Retriever
retriever = vector_db.as_retriever()


# Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# User Input
query = st.text_input("Ask a question")


# Generate Answer
if st.button("Get Answer"):

    if query:

        response = qa_chain.invoke({"query": query})

        st.subheader("Answer")

        st.write(response["result"])