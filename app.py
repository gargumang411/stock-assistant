__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import numpy as np
import re
import time
from typing import List, Dict, Tuple
from langchain.schema import Document
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from langsmith import Client
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_community.tools.tavily_search import TavilySearchResults
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "stock-analysis rag project"
ls_client = Client()

# API configurations
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["ALPHA_VANTAGE_API_KEY"] = os.getenv("ALPHA_VANTAGE_API_KEY")

# Embeddings (defined globally)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Download HuggingFace dataset
@st.cache_resource
def load_vectorstore():
    try:
        dataset_path = snapshot_download(
            repo_id="gargumang411/company_vectors",
            repo_type="dataset",
            cache_dir="./company_vectors_cache"
        )
        vectorstore = Chroma(
            persist_directory=dataset_path,
            embedding_function=embedding_model
        )
        st.success("Loaded vectorstore from HuggingFace dataset.")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vectorstore: {str(e)}")
        raise e

# Initialize vectorstore with error handling
try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error("Application failed to start due to vectorstore loading error. Check logs for details.")
    st.stop()

# LLM setup
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Prompt Templates
qa_prompt = PromptTemplate.from_template("""
You are a financial assistant. Below is the context, including intermediate steps (query variants, retrieved documents, summaries) from web sources, a company database, and Alpha Vantage API. Answer the user's question clearly and concisely, citing sources (Company DB, Web, Alpha Vantage) where relevant. For news queries, prioritize recent events (e.g., 2025). For analysis queries, include key financial metrics (e.g., P/E, EPS, revenue) if available. If data is limited, provide a general response.

Intermediate Steps:
- Query Variants: {query_variants}
- Retrieved Documents: {doc_summaries}

Context:
{context}

Question: {question}
Answer:
""")

summarize_prompt = PromptTemplate.from_template("""
Summarize the following financial document to ~200 words, focusing on recent news, financial metrics (e.g., P/E, EPS, revenue), or analyst recommendations. Exclude irrelevant details (e.g., ads, unrelated companies):

{document}
Summary:
""")

# Helper Functions
def clean_query(query: str) -> str:
    query = re.sub(r'[^\w\s\-\.]', '', query)
    query = " ".join(query.split()[:20])
    return query.strip()

def build_ticker_map(vectorstore) -> Dict[str, str]:
    ticker_map = {}
    docs = vectorstore.get()
    for metadata in docs.get("metadatas", []):
        if metadata.get("ticker") and metadata.get("company_name"):
            ticker_map[metadata["company_name"].lower()] = metadata["ticker"]
    return ticker_map

def extract_ticker(query: str, vectorstore, ticker_map: Dict[str, str]) -> str:
    query_lower = query.lower()
    for company, ticker in ticker_map.items():
        if company in query_lower:
            return ticker
    search_results = vectorstore.similarity_search(query, k=1)
    if search_results and "ticker" in search_results[0].metadata:
        return search_results[0].metadata["ticker"]
    prompt = f"Extract the stock ticker from this query, return only the ticker symbol or 'TSLA' if unclear:\n\"{query}\""
    return llm.invoke(prompt).content.strip()

extract_ticker_lambda = RunnableLambda(
    lambda inputs: extract_ticker(inputs["query"], inputs["vectorstore"], inputs["ticker_map"])
)

def translate_query(query: str, ticker: str) -> str:
    prompt = f"""
You're optimizing user queries for better retrieval in a stock research assistant.

Example:
Original: "any latest news on SMCI?"
Improved: "SMCI latest news 2025 earnings stock performance"

Rewrite the user's query to maximize document retrieval relevance for {ticker}, focusing on financial news or metrics:
\"{query}\"
"""
    return clean_query(llm.invoke(prompt).content.strip())

translate_query_lambda = RunnableLambda(
    lambda inputs: translate_query(inputs["query"], inputs["ticker"])
)

def generate_query_variants(query: str, n=2) -> List[str]:
    prompt = f"Generate {n} alternative phrasings for this financial query, focusing on recent news or financial metrics:\n\n\"{query}\""
    raw = llm.invoke(prompt).content
    variants = [clean_query(line.strip("-• ").strip()) for line in raw.split("\n") if line.strip()]
    return variants[:n]

generate_query_variants_lambda = RunnableLambda(
    lambda inputs: generate_query_variants(inputs["translated_query"], n=2)
)

def fetch_alpha_vantage_data(ticker: str) -> Dict:
    for attempt in range(2):
        try:
            ts = TimeSeries(key=os.environ["ALPHA_VANTAGE_API_KEY"], output_format='json')
            fd = FundamentalData(key=os.environ["ALPHA_VANTAGE_API_KEY"], output_format='json')
            quote_data, _ = ts.get_quote_endpoint(symbol=ticker)
            overview, _ = fd.get_company_overview(symbol=ticker)
            return {
                "price": quote_data.get("05. price"),
                "pe_ratio": overview.get("PERatio"),
                "eps": overview.get("EPS"),
                "revenue": overview.get("RevenueTTM"),
                "market_cap": overview.get("MarketCapitalization")
            }
        except Exception as e:
            print(f"❌ Alpha Vantage API call failed for {ticker} (attempt {attempt+1}): {e}")
            if "rate limit" in str(e).lower():
                time.sleep(5)
            else:
                time.sleep(1)
    return {}

fetch_alpha_vantage_lambda = RunnableLambda(
    lambda inputs: fetch_alpha_vantage_data(inputs["ticker"])
)

def summarize_document(content: str) -> str:
    prompt = summarize_prompt.format(document=content)
    summary = llm.invoke(prompt).content.strip()
    return " ".join(summary.split()[:200])

summarize_document_lambda = RunnableLambda(summarize_document)

def retrieve_docs_with_fusion(query: str, ticker: str, av_data: Dict, k=3) -> Tuple[List[Document], List[str], str]:
    internal_docs = []
    results = vectorstore.similarity_search_with_score(query, k=2)
    for doc, score in results:
        internal_docs.append((doc, score + 0.2))

    web_docs = []
    translated = translate_query(query, ticker)
    variants = generate_query_variants(translated, n=2)
    all_web_queries = [translated] + variants
    query_emb = np.array(embedding_model.embed_query(query))

    for q in all_web_queries:
        for attempt in range(2):
            try:
                search = TavilySearchResults(max_results=2, api_key=os.environ["TAVILY_API_KEY"])
                web_results = search.invoke({"query": q})
                if not isinstance(web_results, list):
                    print(f"⚠️ Tavily returned non-list response for query '{q}':", web_results)
                    continue
                for result in web_results:
                    if isinstance(result, dict) and "content" in result and "url" in result:
                        content = " ".join(result["content"].split()[:800])
                        summarized = summarize_document(content)
                        doc = Document(page_content=summarized, metadata={"source": result["url"]})
                        doc_emb = np.array(embedding_model.embed_documents([summarized])[0])
                        distance = np.linalg.norm(query_emb - doc_emb)
                        web_docs.append((doc, distance))
                break
            except Exception as e:
                print(f"❌ Tavily API call failed for query '{q}' (attempt {attempt+1}): {e}")
                if "rate limit" in str(e).lower():
                    time.sleep(2)
                else:
                    time.sleep(1)

    all_docs = internal_docs + web_docs
    if av_data:
        av_content = f"Alpha Vantage Data for {ticker}: Price: {av_data['price']}, P/E: {av_data['pe_ratio']}, EPS: {av_data['eps']}, Revenue: {av_data['revenue']}, Market Cap: {av_data['market_cap']}"
        av_doc = Document(page_content=av_content, metadata={"source": "Alpha Vantage"})
        all_docs.append((av_doc, 0.0))

    financial_keywords = ["earnings", "revenue", "eps", "p/e", "valuation", "analyst", "rating", "buy", "sell", "news"]
    ranked_docs = []
    for doc, score in all_docs:
        keyword_score = sum(1 for kw in financial_keywords if kw in doc.page_content.lower())
        adjusted_score = score - (keyword_score * 0.1)
        ranked_docs.append((doc, adjusted_score))

    ranked_docs_sorted = sorted(ranked_docs, key=lambda x: x[1])
    top_docs = [doc for doc, _ in ranked_docs_sorted[:k]]
    doc_summaries = "\n".join([f"Source: {doc.metadata.get('source', 'Company DB')}\n{doc.page_content[:100]}..." for doc in top_docs])
    return top_docs, all_web_queries, doc_summaries

retrieve_docs_lambda = RunnableLambda(
    lambda inputs: retrieve_docs_with_fusion(inputs["query"], inputs["ticker"], inputs["av_data"], k=3)
)

# FusionRAG Class
class FusionRAG:
    def __init__(self, llm, vectorstore, docs_per_query=3):
        self.llm = llm
        self.vectorstore = vectorstore
        self.docs_per_query = docs_per_query
        self.ticker_map = build_ticker_map(vectorstore)
        # Cache dataset UUID
        dataset_name = "StockRAG"
        existing_datasets = ls_client.list_datasets()
        self.dataset_id = None
        for dataset in existing_datasets:
            if dataset.name == dataset_name:
                self.dataset_id = dataset.id
                break
        if self.dataset_id is None:
            dataset = ls_client.create_dataset(dataset_name=dataset_name)
            self.dataset_id = dataset.id

        self.pipeline = RunnableSequence(
            lambda inputs: {
                **inputs,
                "ticker": extract_ticker_lambda.invoke({
                    "query": inputs["query"],
                    "vectorstore": self.vectorstore,
                    "ticker_map": self.ticker_map
                })
            },
            lambda inputs: {
                **inputs,
                "translated_query": translate_query_lambda.invoke({
                    "query": inputs["query"],
                    "ticker": inputs["ticker"]
                })
            },
            lambda inputs: {
                **inputs,
                "query_variants": generate_query_variants_lambda.invoke({
                    "translated_query": inputs["translated_query"]
                })
            },
            lambda inputs: {
                **inputs,
                "av_data": fetch_alpha_vantage_lambda.invoke({
                    "ticker": inputs["ticker"]
                })
            },
            lambda inputs: {
                **inputs,
                "docs": retrieve_docs_lambda.invoke({
                    "query": inputs["query"],
                    "ticker": inputs["ticker"],
                    "av_data": inputs["av_data"]
                })[0],
                "query_variants": retrieve_docs_lambda.invoke({
                    "query": inputs["query"],
                    "ticker": inputs["ticker"],
                    "av_data": inputs["av_data"]
                })[1],
                "doc_summaries": retrieve_docs_lambda.invoke({
                    "query": inputs["query"],
                    "ticker": inputs["ticker"],
                    "av_data": inputs["av_data"]
                })[2]
            },
            lambda inputs: {
                "result": self.llm.invoke(qa_prompt.format(
                    context="\n\n".join([f"Source: {'Web' if 'source' in doc.metadata else 'Company DB'}\n{doc.page_content}" for doc in inputs["docs"]]),
                    query_variants=str(inputs["query_variants"]),
                    doc_summaries=inputs["doc_summaries"],
                    question=inputs["query"]
                )).content.strip(),
                "query_variants": inputs["query_variants"],
                "doc_summaries": inputs["doc_summaries"]
            }
        )

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        query = inputs["query"]
        result = self.pipeline.invoke({"query": query})
        ls_client.create_examples(
            dataset_id=self.dataset_id,
            inputs=[{"query": query}],
            outputs=[{"answer": result["result"]}]
        )
        return result

# Cache the pipeline
@st.cache_resource
def get_pipeline():
    return FusionRAG(llm=llm, vectorstore=vectorstore)

# Streamlit UI
st.title("StockRAG: Financial Analysis Assistant")
st.markdown("""
Enter a financial query (e.g., 'Fundamental Analysis of SMCI' or 'any latest news on smci?') to get detailed insights from company profiles, web news, and financial metrics.
""")

with st.form("query_form"):
    query = st.text_input("Enter your query:", placeholder="e.g., Latest news about SMCI earnings")
    submit = st.form_submit_button("Submit")

if submit and query.strip():
    with st.spinner("Fetching answer..."):
        try:
            qa_chain = get_pipeline()
            response = qa_chain.invoke({"query": query})
            st.markdown("### Answer")
            st.markdown(response["result"])
            with st.expander("Query Variants"):
                st.write(response["query_variants"])
            with st.expander("Retrieved Documents"):
                st.write(response["doc_summaries"])
        except Exception as e:
            st.error(f"Error during query processing: {str(e)}")
