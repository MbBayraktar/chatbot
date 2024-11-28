import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from streamlit.components.v1 import html
from dotenv import load_dotenv

load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="Techpro Chatbot", page_icon="💻")

# Load specific data from Excel
def load_specific_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Örneğin sadece 'Soru' ve 'Cevap' sütunlarını alalım
    documents = []
    # Her satırdaki Soru ve Cevap'ı birleştirerek anlamlı bir bağlam oluşturma
    for _, row in df.iterrows():
        if pd.notna(row['Questions']) and pd.notna(row['Answers']):
            # Soru ve cevabı birleştirerek içerik oluşturuyoruz
            content = f"Soru: {row['Questions']}\nCevap: {row['Answers'].casefold()}"
            documents.append(Document(page_content=content))
    
    return documents

file_path = "combined_questions_answers.xlsx"
documents = load_specific_data(file_path)

# Text splitting
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
#texts = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2", 
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
    )

persist_directory = 'db'  
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory
    )

retriever = vectordb.as_retriever()


# Streamlit UI for Chat with enhanced visuals
st.markdown("""
    <style>
        .main-header {
             display: flex;
        justify-content: center; /* Metni yatay olarak ortaya yerleştirir */
        align-items: center; /* Metni dikey olarak ortaya yerleştirir */
        background-color: #f1f1f1;
        padding: 10px;
        color: #228B22; /* Açık yeşil renk */
        font-weight: bold; /* Yazıyı belirginleştirmek için opsiyonel */
        }
        .main-header img {
            height: 50px;
        }
        .main-header h1 {
            font-size: 24px;
            text-align: center;
            flex-grow: 1;
        }
        .hide-bar {
            display: none;
        }
        .chat-container {
            background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .chat-message-user {
            background-color: #D1ECF1;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 5px;
        }
        .chat-message-assistant {
            background-color: #FFF3CD;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 5px;
        }
        body {
            background-image: url('https://media.licdn.com/dms/image/v2/D4D0BAQGRmZ5oGHzj8w/company-logo_200_200/company-logo_200_200/0/1689611269635/techpro_education_tr_logo?e=1740009600&v=beta&t=9IxoLv6JxZohhAdXHq9f9DeycfZJcGB8ynqsfShMh-s');
            background-size: contain;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 0.9; /* Soft look */
        }
    </style>
""", unsafe_allow_html=True)

# Header with logos and Data Scientist Group title
st.markdown("""
    <div class="main-header">    
        <h1>TechPro Data Science Group</h1>
    </div>
""", unsafe_allow_html=True)
# Chatbot functionality remains the same...

# Add Chatbot GIF in bottom-right corner

chatbot_gif_html = """
    <div style="position: fixed; bottom: 60px; right: 60px; z-index: 1000;">
        <a href="https://www.techproeducation.com.tr/kurs/data-science" target="_blank" style="position: relative;">
            <img src="https://miro.medium.com/v2/resize:fit:1400/1*fZsdZisozTZbM6AaPQKI4Q.gif" alt="Chatbot GIF" 
                 style="width: 150px; height: 150px; border-radius: 80%;">
            <span style="position: absolute; top: -70px; left: 50%; transform: translateX(-50%); color: #006400; font-size: 18px; font-weight: bold; background-color: rgba(255, 255, 255, 0.6); padding: 2px 8px; border-radius: 5px;">Join Us</span>
        </a>
    </div>
"""
st.markdown(chatbot_gif_html, unsafe_allow_html=True)

# Hide the Streamlit sidebar
st.markdown('<style>.sidebar {display: none;}</style>', unsafe_allow_html=True)

st.header("Chat with Techpro Mentoring 💬")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Data Science!"}]

# Wrap the prompt in a function

def prompt_fn(query: str, context: str) -> str:
    return f"""
    You are a Data Science Instructor and Mentor. 
    If the user's query matches any question from the database, return the corresponding answer directly.
    Otherwise, answer the user's question using the information from the context below and up to 3 sentences your answer. 
    If you don't find the answer in the context, respond with "Bu konu hakkında bilgi sahibi değilim, Lütfen Data Science/Mentroing hakkında 
    soru sorun."
    Context: {context} 
    User's question: {query}"""

# LLM model
@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=150)
llm = load_llm()

@st.cache_resource
def create_rag_chain():
    from langchain_core.runnables import RunnableLambda
    prompt_runnable = RunnableLambda(lambda inputs: prompt_fn(inputs["query"], inputs["context"]))
    return prompt_runnable | llm | StrOutputParser()

rag_chain = create_rag_chain()

def generate_response(query):
   
    results = retriever.get_relevant_documents(query)[:3]  
    context = "\n".join([doc.page_content for doc in results])
    inputs = {"query": query, "context": context}
    response = rag_chain.invoke(inputs)

    return response

USER_ICON_URL = "https://cdn-icons-png.flaticon.com/512/8635/8635572.png"
ASSISTANT_ICON_URL = "https://media.licdn.com/dms/image/v2/D4D0BAQGRmZ5oGHzj8w/company-logo_200_200/company-logo_200_200/0/1689611269635/techpro_education_tr_logo?e=1740009600&v=beta&t=9IxoLv6JxZohhAdXHq9f9DeycfZJcGB8ynqsfShMh-s"

# Display chat messages from session state with custom icons
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 10px;">
            <img src="{USER_ICON_URL}" alt="User Icon" style="width: 50px; height: 50px; border-radius: 25%; margin-right: 10px;">
            <div style="background-color: #D1ECF1; border-radius: 10px; padding: 10px; flex-grow: 1;">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif message["role"] == "assistant":
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 10px;">
            <img src="{ASSISTANT_ICON_URL}" alt="Assistant Icon" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 10px;">
            <div style="background-color: #FFF3CD; border-radius: 10px; padding: 10px; flex-grow: 1;">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Handle user query input with custom user icon
if query := st.chat_input("Your question"):
    st.session_state["messages"].append({"role": "user", "content": query})
    st.markdown(f"""
    <div style="display: flex; align-items: flex-start; margin-bottom: 10px;">
        <img src="{USER_ICON_URL}" alt="User Icon" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 10px;">
        <div style="background-color: #D1ECF1; border-radius: 10px; padding: 10px; flex-grow: 1;">
            {query}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Generate response with custom assistant icon
    with st.spinner("Thinking..."):
        response = generate_response(query)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 10px;">
            <img src="{ASSISTANT_ICON_URL}" alt="Assistant Icon" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 10px;">
            <div style="background-color: #FFF3CD; border-radius: 10px; padding: 10px; flex-grow: 1;">
                {response}
            </div>
        </div>
        """, unsafe_allow_html=True)