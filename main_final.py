import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()

# --- VALIDATION: Check Key in both .env (Local) and Secrets (Cloud) ---
api_key = os.getenv("GOOGLE_API_KEY")

# If no local key, check Streamlit Cloud Secrets
if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        pass

# Final check
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please add it to .env (local) or Secrets (cloud).")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="DocuQuery: Final Version", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    # Local Embeddings (Free)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # --- THE FIX: Using the exact model from your allowed list ---
    print("DEBUG: Initializing ChatGoogleGenerativeAI with model=gemini-exp-1206")
    llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.3,
    google_api_key=api_key
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first!")
        return

    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for message in st.session_state.chat_history:
            if message.type == "human":
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)
    except Exception as e:
        st.error(f"Error during generation: {e}")

def main():
    st.title(" DocuQuery: A RAG Based Application ")
    st.markdown("Powered by **Local Embeddings** + **Gemini Exp 1206**")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload a PDF first.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    # This resets the memory with the NEW model
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("AI Ready! Ask your question.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.chat_input("Ask a question...")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
