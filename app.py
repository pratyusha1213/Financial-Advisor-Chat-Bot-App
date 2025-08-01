import streamlit as st
import time
import random
import json
import os 
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pyrebase
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool # Import the base Tool class
from langchain_core.messages import HumanMessage, AIMessage
from streamlit.errors import StreamlitAPIException, StreamlitSecretNotFoundError

# Import all necessary functions from utils
from utils import (
    create_vector_store, # Import the function to build the DB
    get_retriever,
    get_compression_retriever,
    get_multi_query_retriever,
    get_current_stock_price,
    get_company_info,
    calculate_investment_projection,
    update_knowledge_base
)

# --- FIREBASE & APP CONFIGURATION (Unchanged) ---
load_dotenv()
FIREBASE_CONFIG = {
  "apiKey": os.getenv("FIREBASE_API_KEY"), "authDomain": os.getenv("AUTH_DOMAIN"),
  "projectId": os.getenv("PROJECT_ID"), "storageBucket": os.getenv("STORAGE_BUCKET"),
  "messagingSenderId": os.getenv("MESSAGING_SENDER_ID"), "appId": os.getenv("APP_ID"),
  "databaseURL": os.getenv("DATABASE_URL")
}
if not FIREBASE_CONFIG["apiKey"] or not FIREBASE_CONFIG["databaseURL"]:
    st.error("Firebase configuration not found. Please ensure your .env file is correctly populated.")
    st.stop()

# --- FIREBASE INITIALIZATION (Unchanged) ---
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            try:
                cred_info = st.secrets.get("firebase_credentials")
                cred = credentials.Certificate(dict(cred_info))
            except (StreamlitAPIException, StreamlitSecretNotFoundError):
                cred = credentials.Certificate("firebase_credentials.json")
            firebase_admin.initialize_app(cred)
        return pyrebase.initialize_app(FIREBASE_CONFIG)
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        return None
firebase = initialize_firebase()
db = firestore.client() if firebase else None

# --- AUTHENTICATION PAGES (Unchanged) ---
def signup_page():
    st.header("Create a New Account")
    name = st.text_input("Name", key="signup_name")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    if st.button("Sign Up", use_container_width=True):
        if not all([name, email, password]):
            st.error("Please fill in all fields.")
            return
        try:
            user = firebase.auth().create_user_with_email_and_password(email, password)
            db.collection('users').document(user['localId']).set({'name': name})
            st.success("Account created successfully! Please go to the Login tab.")
        except Exception as e: st.error(f"Could not create account: {e}")

def login_page():
    st.header("Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", use_container_width=True):
        try:
            user = firebase.auth().sign_in_with_email_and_password(email, password)
            uid = user['localId']
            user_info = db.collection('users').document(uid).get()
            st.session_state.logged_in, st.session_state.uid, st.session_state.user_name = True, uid, user_info.to_dict().get('name', 'User')
            if uid not in st.session_state.user_conversations: st.session_state.user_conversations[uid] = {}
            st.rerun()
        except Exception: st.error("Login failed. Please check your email and password.")

def forgot_password_page():
    st.header("Reset Your Password")
    email = st.text_input("Enter your account email", key="forgot_email")
    if st.button("Send Reset Link", use_container_width=True):
        if not email:
            st.error("Please enter your email address.")
            return
        try:
            firebase.auth().send_password_reset_email(email)
            st.success("A password reset link has been sent to your email address.")
        except Exception as e: st.error(f"Could not send reset email: {e}")

# --- AGENT INITIALIZATION (Unchanged) ---
def initialize_agent():
    """Initializes the agent components, building the knowledge base if it doesn't exist."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    if 'rag_strategy' not in st.session_state:
        st.session_state.rag_strategy = random.choice(['standard', 'compressed', 'multi_query'])
    strategy = st.session_state.rag_strategy
    print(f"--- Session assigned to RAG Strategy: {strategy.upper()} ---")

    if not os.path.exists("chroma_db"):
        with st.spinner("Building knowledge base for the first time... This may take a minute."):
            try:
                create_vector_store(data_path="knowledge_base")
                st.success("Knowledge base built successfully!")
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")
                st.session_state.agent_ready = False
                return

    try:
        if strategy == 'standard': retriever = get_retriever()
        elif strategy == 'compressed': retriever = get_compression_retriever(llm=llm)
        else: retriever = get_multi_query_retriever(llm=llm)

        def retrieve_and_format_docs(query: str) -> str:
            docs = retriever.invoke(query)
            if not docs: return "No information found in the knowledge base for this query."
            return "\n\n".join(
                f"Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
                for doc in docs
            )

        retriever_tool = Tool(
            name="financial_knowledge_search",
            func=retrieve_and_format_docs,
            description="Search for financial concepts, investment strategies, and market analysis from the knowledge base. Provides content and sources."
        )

        tools = [retriever_tool, get_current_stock_price, get_company_info, calculate_investment_projection]
    except FileNotFoundError as e:
        st.error(f"Failed to initialize knowledge base: {e}", icon="🚨")
        st.session_state.agent_ready = False
        return
        
    st.session_state.llm = llm
    st.session_state.tools = tools
    st.session_state.agent_ready = True
    print("Agent components initialized successfully.")

# --- MAIN APP ---
def get_chat_title(messages):
    if messages and isinstance(messages[0], HumanMessage):
        title = messages[0].content.split('\n')[0]
        return title[:30] + '...' if len(title) > 30 else title
    return "New Chat"

def main_app():
    st.title(f"💰 Welcome, {st.session_state.user_name}!")
    st.caption("Your personalized AI guide for financial queries.")

    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.user_name}**")
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.active_chat_id = None
            st.rerun()
        st.markdown("---")
        st.subheader("Your Conversations")
        user_chats = st.session_state.user_conversations.get(st.session_state.uid, {})
        sorted_chat_ids = sorted(user_chats.keys(), reverse=True)
        chat_list_container = st.container()
        with chat_list_container:
            for chat_id in sorted_chat_ids:
                if not user_chats[chat_id]: continue
                chat_title = get_chat_title(user_chats[chat_id])
                if st.button(chat_title, key=chat_id, use_container_width=True):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()
        st.markdown("---")
        st.subheader("Knowledge Base")
        if st.button("Update from Web", use_container_width=True):
            with st.spinner("Fetching new articles..."):
                update_status = update_knowledge_base()
                st.success(update_status)
                if 'agent_ready' in st.session_state: del st.session_state.agent_ready
                st.rerun()
        strategy_map = {'standard': 'Standard (Fast)', 'compressed': 'Compressed (Precise)', 'multi_query': 'Multi-Query (Robust)'}
        strategy_name = strategy_map.get(st.session_state.rag_strategy, "Unknown")
        st.caption(f"🧪 RAG Strategy: **{strategy_name}**")
        if st.button("Logout", use_container_width=True, type="secondary"):
            keys_to_delete = ['logged_in', 'uid', 'user_name', 'active_chat_id']
            for key in keys_to_delete:
                # --- THIS IS THE FIX for the logout error ---
                # Changed del st.session_state.key to del st.session_state[key]
                if key in st.session_state: del st.session_state[key]
            st.rerun()

    if 'active_chat_id' not in st.session_state or st.session_state.active_chat_id is None:
        st.session_state.active_chat_id = f"chat_{time.time()}"
        st.session_state.user_conversations[st.session_state.uid][st.session_state.active_chat_id] = []
    active_chat_id = st.session_state.active_chat_id
    chat_history = st.session_state.user_conversations[st.session_state.uid][active_chat_id]

    for message in chat_history:
        with st.chat_message(message.type): st.markdown(message.content)

    if user_input := st.chat_input("Ask a financial question..."):
        st.chat_message("user").markdown(user_input)
        chat_history.append(HumanMessage(content=user_input))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a specialized Financial Advisor Bot assisting {st.session_state.user_name}. You must only use the tools provided to you. If you cannot answer using your tools, politely say you cannot help. When you use the 'financial_knowledge_search' tool, you MUST cite the source and page number for the information in your final answer. For example: 'According to [Source File], page [Page Number], dollar-cost averaging is...'."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(st.session_state.llm, st.session_state.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=st.session_state.tools, verbose=True)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
                    answer = response["output"]
                    st.markdown(answer)
                    chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    error_message = f"Sorry, an error occurred: {e}"
                    st.error(error_message)
                    chat_history.append(AIMessage(content=error_message))
        if len(chat_history) == 2: st.rerun()

# --- MAIN APP ROUTER ---
if "agent_ready" not in st.session_state: initialize_agent()
if "user_conversations" not in st.session_state: st.session_state.user_conversations = {}
if firebase and st.session_state.get("agent_ready", False):
    if st.session_state.get("logged_in", False): main_app()
    else:
        # --- THIS IS THE FIX for the RAG strategy error ---
        # The line that deleted 'rag_strategy' has been removed.
        # The strategy will now persist correctly across logins/logouts in the same session.
        login_tab, signup_tab, forgot_password_tab = st.tabs(["Login", "Sign Up", "Forgot Password"])
        with login_tab: login_page()
        with signup_tab: signup_page()
        with forgot_password_tab: forgot_password_page()
elif not firebase: st.error("Firebase connection failed. The app cannot start.")
else: st.warning("Application is initializing. Please wait.")
