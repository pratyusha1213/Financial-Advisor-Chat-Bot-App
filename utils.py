__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import re 
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import yfinance as yf
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables from the .env file
load_dotenv()

# --- WEB SCRAPING & KNOWLEDGE BASE UPDATE (Unchanged) ---
def scrape_reuters_financial_news(num_articles: int = 5):
    """Scrapes the latest financial news articles from Reuters."""
    print("--- Calling tool: scrape_reuters_financial_news ---")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    url = "https://www.reuters.com/business/finance/"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        links = soup.find_all('a', attrs={'data-testid': 'Heading'}, limit=num_articles * 2)
        for link in links:
            if len(articles) >= num_articles: break
            article_url = "https://www.reuters.com" + link['href']
            try:
                article_response = requests.get(article_url, headers=headers)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                paragraphs = article_soup.find_all('p', class_='text__text__1FZLe')
                content = "\n".join([p.get_text() for p in paragraphs])
                if content: articles.append(Document(page_content=content, metadata={"source": article_url}))
            except Exception as e: print(f"Could not scrape article {article_url}: {e}")
        return articles
    except Exception as e:
        print(f"Failed to scrape Reuters homepage: {e}")
        return []

def update_knowledge_base(index_path: str = "chroma_db"):
    """Scrapes new articles and adds them to the existing ChromaDB vector store."""
    print("--- Starting Knowledge Base Update ---")
    new_docs = scrape_reuters_financial_news()
    if not new_docs: return "No new articles found or failed to scrape."
    if not os.path.exists(index_path): return "Error: ChromaDB not found."
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectordb = Chroma(persist_directory=index_path, embedding_function=embeddings)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        doc_chunks = text_splitter.split_documents(new_docs)
        vectordb.add_documents(doc_chunks)
        vectordb.persist()
        return f"Successfully added {len(new_docs)} new articles to the knowledge base."
    except Exception as e: return f"An error occurred while updating the knowledge base: {e}"

# --- KNOWLEDGE BASE CREATION & RETRIEVAL (Unchanged) ---
def create_vector_store(data_path: str = "knowledge_base/", index_path: str = "chroma_db"):
    """Builds and saves a ChromaDB vector store from PDF documents."""
    if os.path.exists(index_path):
        print(f"ChromaDB at '{index_path}' already exists. Skipping creation.")
        return
    print(f"Creating new ChromaDB from documents in '{data_path}'...")
    if not os.listdir(data_path): raise ValueError(f"No files found in '{data_path}'.")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    if not documents: raise ValueError(f"Could not load any documents from '{data_path}'.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=index_path)
    vectordb.persist()
    print(f"ChromaDB created and saved at '{index_path}'.")

def get_retriever(index_path: str = "chroma_db"):
    """Loads an existing ChromaDB and returns it as a standard retriever."""
    if not os.path.exists(index_path): raise FileNotFoundError(f"ChromaDB not found at '{index_path}'.")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(persist_directory=index_path, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 4})

def get_compression_retriever(llm: ChatOpenAI, index_path: str = "chroma_db"):
    """Creates a contextual compression retriever."""
    base_retriever = get_retriever(index_path)
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

def get_multi_query_retriever(llm: ChatOpenAI, index_path: str = "chroma_db"):
    """Creates a multi-query retriever."""
    base_retriever = get_retriever(index_path)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    return multi_query_retriever

# --- FINANCIAL TOOLS (Updated with Input Validation) ---
@tool
def get_current_stock_price(ticker: str) -> str:
    """Fetches the current stock price for a given ticker symbol."""
    # --- ADDED VALIDATION ---
    if not re.match(r"^[A-Z]{1,5}$", ticker.upper()):
        return "Error: Invalid ticker symbol format. Please use 1-5 uppercase letters (e.g., 'AAPL')."
    
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('regularMarketPrice')
        if price: return f"The current price of {ticker.upper()} is ${price:.2f}."
        else: return f"Could not retrieve the current price for {ticker.upper()}."
    except Exception: return f"Error fetching stock price for {ticker}."

@tool
def get_company_info(ticker: str) -> str:
    """Provides a summary of a company's business for a given ticker symbol."""
    # --- ADDED VALIDATION ---
    if not re.match(r"^[A-Z]{1,5}$", ticker.upper()):
        return "Error: Invalid ticker symbol format. Please use 1-5 uppercase letters (e.g., 'MSFT')."
        
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info.get('longBusinessSummary'): return f"No detailed information for ticker {ticker}."
        summary, sector, market_cap, name = info.get('longBusinessSummary', 'N/A'), info.get('sector', 'N/A'), info.get('marketCap', 0), info.get('longName', ticker.upper())
        return (f"**{name} ({ticker.upper()})**\n**Sector:** {sector}\n**Market Cap:** ${market_cap:,}\n\n**Business Summary:**\n{summary}")
    except Exception: return f"Error fetching company info for {ticker}."

@tool
def calculate_investment_projection(principal: float, monthly_contribution: float, years: int, annual_rate: float) -> str:
    """
    Calculates the future value of an investment. 'annual_rate' should be a decimal (e.g., 7% is 0.07).
    Returns a JSON string with the projection details.
    """
    print(f"--- Calling tool: calculate_investment_projection ---")
    monthly_rate, months = annual_rate / 12, years * 12
    fv_principal = principal * ((1 + monthly_rate) ** months)
    fv_contributions = monthly_contribution * ((((1 + monthly_rate) ** months) - 1) / monthly_rate) if monthly_rate > 0 else monthly_contribution * months
    total_fv = fv_principal + fv_contributions

    projection_data = {
        "initial_principal": f"${principal:,.2f}",
        "monthly_contribution": f"${monthly_contribution:,.2f}",
        "investment_period_years": years,
        "estimated_annual_rate_percent": f"{annual_rate*100:.1f}%",
        "projected_future_value": f"${total_fv:,.2f}"
    }
    
    return json.dumps(projection_data, indent=2)

if __name__ == '__main__':
    try:
        create_vector_store()
    except ValueError as e:
        print(f"Error: {e}")
