# 💰 Financial Advisor Bot

This project is an advanced, AI-powered Financial Advisor Bot built with **Streamlit** and **LangChain**. It showcases a sophisticated agentic architecture that combines multiple advanced RAG strategies with live function calling, secure user authentication, a multi-chat interface, and automated knowledge base updates from the web.

This application serves as a comprehensive demonstration of how to build, deploy, and scale a dynamic, stateful, and production-ready AI assistant.

---

## ✨ Key Features

### Advanced RAG Strategies

The bot’s foundational knowledge is built from a custom library of financial documents (PDFs) using a **ChromaDB** vector store.

- **A/B Testing:** Each user session randomly utilizes one of three RAG strategies:
  - **Standard:** A fast, baseline retriever.
  - **Contextual Compression:** Retrieves documents and uses an LLM to extract the most relevant sentences for high-precision responses.
  - **Multi-Query (Query Translation):** Rewrites the user query into multiple variations for broader, more robust searching.

- **Source Citation:** The bot cites the exact document and page number used to answer queries.

### Live Function Calling

The agent can fetch real-time financial data through API-integrated tools:
- `get_current_stock_price`: Fetches live stock prices via Yahoo Finance.
- `get_company_info`: Retrieves business summaries for public companies.
- `calculate_investment_projection`: Calculates future investment values.

**Input Validation:** Tools validate stock ticker format before making API calls.

### User Experience & Interface

- **Secure User Authentication:** Implemented with Firebase Authentication (Email/Password).
- **Personalization:** Bot greets users by name, with persistent chat history saved per user in Firestore.
- **Multi-Chat Interface:** Inspired by modern chat tools, supports multiple conversations with auto-titling and history navigation.
- **Automated Knowledge Base Updates:**  
  Users can click the “Update from Web” button to fetch and integrate the latest financial news from **Reuters** without restarting the app.

---

## 🛠️ Tech Stack & Architecture

- **Frontend:** Streamlit  
- **AI Orchestration:** LangChain  
- **Language Model (LLM):** OpenAI GPT-4o  
- **Vector Database:** ChromaDB  
- **User Management:** Firebase Authentication & Firestore  
- **Data Fetching:** `yfinance`, `requests`, `BeautifulSoup4`

The app uses a **LangChain Agent** that decides whether to:
- Search the vector store for conceptual questions.
- Call live tools for real-time data.
- Synthesize answers from multiple sources for comprehensive financial advice.

---

## 🚀 Local Setup and Installation

### Prerequisites

- Python 3.9+  
- A GitHub account  
- OpenAI API key with billing enabled  
- A Google Firebase project

### 1. Set Up Firebase

- Create a new project at [Firebase Console](https://console.firebase.google.com).
- Enable:
  - **Authentication** → Email/Password
  - **Firestore Database** → Test Mode
  - **Realtime Database** → Test Mode
- Download your **Admin SDK private key** as `firebase_credentials.json`.
- Copy your **Web App Config** details (used in `.env`).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd financial-advisor-bot
````

### 3. Set Up a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

* Place your downloaded Firebase Admin key in the root directory as:
  `firebase_credentials.json`

* Create a `.env` file and add:

```env
OPENAI_API_KEY="sk-..."
FIREBASE_API_KEY="AIzaSy..."
AUTH_DOMAIN="your-app.firebaseapp.com"
PROJECT_ID="your-app-id"
STORAGE_BUCKET="your-app.appspot.com"
MESSAGING_SENDER_ID="..."
APP_ID="..."
DATABASE_URL="https://your-app-id.firebaseio.com"
```

### 6. Build the Knowledge Base

* Place your financial PDFs in the `knowledge_base/` folder.
* Run the following to build the initial **ChromaDB** index:

```bash
python utils.py
```

### 7. Run the Application

```bash
streamlit run app.py
```

---

## ✅ Summary

This project demonstrates how to combine cutting-edge LLM capabilities with real-time data, secure authentication, and interactive UI elements to build a powerful financial assistant. Ideal for learning how to scale intelligent agents in real-world applications.