import os
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


# Load PDF
loader = PyPDFLoader("data/Gudz_FullStack_AI_Intern_JD.pdf")

documents = loader.load()

print("PDF Loaded")


# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

print("Text Split Done")


# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings Ready")


# Create ChromaDB
vector_db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="chroma_db"
)

vector_db.persist()

print("ChromaDB Created")


# Create retriever
retriever = vector_db.as_retriever()


# Load Groq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("Chatbot Ready!")


# Chat loop
while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    response = qa_chain.invoke({"query": query})

    print("\nAnswer:")
    print(response["result"])