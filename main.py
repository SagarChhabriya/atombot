import json
import os
import streamlit as st
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import shutil


# Load environment variables
load_dotenv()
st.set_page_config(page_title="atomcamp")

# CORPUS_PATH = "tt_train.jsonl"
# CORPUS_PATH = "train.jsonl"
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "train.jsonl")
PERSIST_DIR = "./faiss_868-01"

# Load the corpus
docs = []
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            instruction = item.get("instruction")
            content = item.get("output")
            text = instruction + "\n" + content
            docs.append(text)
        except json.JSONDecodeError as e:
            print(f"Skipping line due to JSON error: {e}")
            continue  # Skip this line if it’s malformed

# Check if FAISS persistence directory exists
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

# Initialize embedding model
# embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # To Prevent PyTorch issues with python 13.3
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001") # To Prevent PyTorch issues with python 13.3

# st.write("length of in memory list containing docs: ", len(docs))
# st.write("before loading faiss")

# Step 1: Load or create FAISS vector store
# Safe FAISS loading with fallback to rebuild
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    try:
        vectordb = FAISS.load_local(PERSIST_DIR, embedding_model, allow_dangerous_deserialization=True)
        print("Loaded FAISS index from disk.")
        print("No of documents in the corpus: ", vectordb.index.ntotal)
        st.write(vectordb.index.ntotal)
        
    except AssertionError as e:
        print(f"FAISS dimension mismatch: {e}. Rebuilding index.")
        shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR)
        vectordb = FAISS.from_texts(texts=docs, embedding=embedding_model)
        vectordb.save_local(PERSIST_DIR)
        print("Rebuilt FAISS index and saved to disk.")
else:
    vectordb = FAISS.from_texts(texts=docs, embedding=embedding_model)
    vectordb.save_local(PERSIST_DIR)
    print("Created FAISS index and saved to disk.")

# st.write("after loading faiss")

# Step 2: Create retriever and chat model
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """
   You are a helpful assistant. Use the following retrieved information to answer the question.
    
    Question: {input}
    
    Documents:
    {context}
    
    Answer:
    """
)

# Step 3: Setup QA chain
document_chain = create_stuff_documents_chain(llm=chat_model, prompt=prompt)
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

# Streamlit UI setup
st.title("Atombob")

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if user_input := st.chat_input("Type a message"):
    # Show user input immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response and show it in real-time
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Generate the bot's response
        response = qa_chain.invoke({"input": user_input})
        bot_response = response['answer']

        # Simulate streaming effect
        for sentence in bot_response.split(". "):
            full_response += sentence + (". " if not sentence.endswith(".") else "")
            response_placeholder.write(full_response.strip())
            time.sleep(0.5)

        # Add bot's final response to session history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response.strip()}
        )
