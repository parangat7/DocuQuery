import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# --- THIS IS THE FIX: Using Local Embeddings (Free & Unlimited) ---
from langchain_huggingface import HuggingFaceEmbeddings
# --- We only use Google for the CHAT, not the embedding ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="DocuQuery: RAG Assistant", layout="wide")

# --- Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Splits the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Converts chunks into vectors and stores them in FAISS."""
    # --- CRITICAL CHANGE ---
    # We use "all-MiniLM-L6-v2". It runs on your laptop.
    # It does NOT send data to Google, so you cannot get a Quota error.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Creates the RAG chain with history."""
    # We are using the specific model found in your allowed list
    llm = ChatGoogleGenerativeAI(model="gemini-exp-1206", temperature=0.3)
    
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
    """Handles the user query and updates chat history."""
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first!")
        return

    # Get response from the chain
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation
    for message in st.session_state.chat_history:
        if message.type == "human":
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

# --- Frontend (Streamlit) ---

def main():
    st.title("📄 DocuQuery: Enterprise RAG Assistant")
    st.markdown("Powered by **Local Embeddings** + **Google Gemini**")

    # Sidebar for uploading PDF
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here", accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload a PDF first.")
            else:
                with st.spinner("Processing... (Downloading model to laptop...)"):
                    try:
                        # 1. Get PDF Text
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text:
                            st.error("Could not extract text. Is this a scanned PDF?")
                            return

                        # 2. Get Text Chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Create Vector Store
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # 4. Create Conversation Chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Done! AI is ready to answer.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Chat Input
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()