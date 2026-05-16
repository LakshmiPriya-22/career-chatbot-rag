# Complete AI Career Chatbot with Modern UI


import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# LOAD ENV VARIABLES
load_dotenv()

# PAGE CONFIG
st.set_page_config(
    page_title="AI Career Chatbot",
    page_icon="🤖",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

/* MAIN TITLE */
h1 {
    color: white;
    text-align: center;
    font-size: 55px;
    font-weight: bold;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #e2e8f0;
    font-size: 22px;
    margin-bottom: 40px;
}

/* FEATURE CARDS */
.feature-card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}

.feature-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 10px;
}

.feature-text {
    color: #cbd5e1;
    margin-top: 10px;
}

/* UPLOAD SECTION */
.upload-box {
    background-color: #1e293b;
    padding: 30px;
    border-radius: 20px;
    margin-top: 30px;
}

/* FILE UPLOADER LABEL */
[data-testid="stFileUploader"] label {
    color: white !important;
    font-size: 18px !important;
    font-weight: bold;
}

/* INPUT LABELS */
label {
    color: white !important;
    font-size: 18px !important;
    font-weight: bold;
}

/* TEXT INPUT */
.stTextInput input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #475569;
    padding: 12px;
}

/* TEXT INPUT PLACEHOLDER */
.stTextInput input::placeholder {
    color: #cbd5e1 !important;
}

/* TEXT AREA */
.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #475569;
    padding: 12px;
}

/* TEXT AREA PLACEHOLDER */
.stTextArea textarea::placeholder {
    color: #cbd5e1 !important;
}

/* BUTTONS */
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* SUBHEADERS */
h2, h3 {
    color: white !important;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 50px;
    color: #94a3b8;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)
# HERO SECTION
st.markdown("""
<h1>🤖 AI Career Chatbot using RAG</h1>
<p class='subtitle'>Upload your resume and get AI-powered career guidance, ATS analysis, interview preparation, and job matching.</p>
""", unsafe_allow_html=True)

# FEATURE CARDS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <div style='font-size:55px;'>📄</div>
        <div class='feature-title'>ATS Analysis</div>
        <div class='feature-text'>Analyze resume quality and ATS score.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <div style='font-size:55px;'>🎯</div>
        <div class='feature-title'>Job Matching</div>
        <div class='feature-text'>Compare resume with job descriptions.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <div style='font-size:55px;'>💬</div>
        <div class='feature-title'>Career Guidance</div>
        <div class='feature-text'>Ask career and interview questions.</div>
    </div>
    """, unsafe_allow_html=True)

# FILE UPLOAD
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stFileUploader"] label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Resume PDF",
    type=["pdf"]
)

st.markdown("</div>", unsafe_allow_html=True)

# MAIN CHATBOT LOGIC
if uploaded_file is not None:

    st.success("✅ Resume Uploaded Successfully")

    # SAVE TEMP FILE
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # LOAD PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # EXTRACT TEXT
    resume_text = ""
    for doc in documents:
        resume_text += doc.page_content

    # TEXT SPLITTING
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    # EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # VECTOR DATABASE
    vector_db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="chroma_db"
    )

    # RETRIEVER
    retriever = vector_db.as_retriever()

    # GROQ MODEL
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    # RAG CHAIN
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    st.divider()

    # CHATBOT SECTION
    st.subheader("💬 Ask Career Questions")

    query = st.text_input(
        "Ask something about your resume or career"
    )

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)

        st.success(response)

    st.divider()

    # ATS ANALYSIS
    st.subheader("📄 ATS Resume Analysis")

    if st.button("Analyze Resume"):

        ats_prompt = f"""
        Analyze this resume professionally.

        Give:
        1. Resume Strengths
        2. Weaknesses
        3. Missing Skills
        4. ATS Score out of 100
        5. Suggestions for Improvement

        Resume:
        {resume_text}
        """

        with st.spinner("Analyzing Resume..."):
            ats_response = llm.invoke(ats_prompt)

        st.write(ats_response.content)

    st.divider()

    # JOB MATCHING
    st.subheader("🎯 Resume Job Matching")

    job_description = st.text_area(
        "Paste Job Description"
    )

    if st.button("Match Resume with Job"):

        match_prompt = f"""
        Compare the following resume with the job description.

        Give:
        1. Match Percentage
        2. Matching Skills
        3. Missing Skills
        4. Improvement Suggestions
        5. Recommended Learning Path

        Resume:
        {resume_text}

        Job Description:
        {job_description}
        """

        with st.spinner("Matching Resume..."):
            match_response = llm.invoke(match_prompt)

        st.write(match_response.content)

    st.divider()

    # INTERVIEW QUESTIONS
    st.subheader("🎤 Interview Questions")

    if st.button("Generate Interview Questions"):

        interview_prompt = f"""
        Based on this resume,
        generate technical and HR interview questions.

        Resume:
        {resume_text}
        """

        with st.spinner("Generating Questions..."):
            interview_response = llm.invoke(interview_prompt)

        st.write(interview_response.content)

# FOOTER
st.markdown("""
<div class='footer'>
Built with ❤️ using LangChain, Groq, ChromaDB, HuggingFace, and Streamlit
</div>
""", unsafe_allow_html=True)