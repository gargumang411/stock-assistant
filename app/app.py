try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except KeyError:
    pass  # fallback to default sqlite3
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
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
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
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# Load the model from the local directory
embedding_model = LocalSentenceTransformerEmbeddings('local_model')

# Download HuggingFace dataset
@st.cache_resource
def load_vectorstore():
    dataset_path = snapshot_download(
        repo_id="gargumang411/company_vectors",
        repo_type="dataset",
        cache_dir="./company_vectors_cache"
    )
    vectorstore = Chroma(
        persist_directory=dataset_path,
        embedding_function=embedding_model
    )
    print("✅ Vectorstore loaded.")  # Still here for console debugging
    return vectorstore

# Initialize vectorstore with error handling
for attempt in range(3):
    try:
        vectorstore = load_vectorstore()
        break
    except Exception as e:
        error_msg = f"Attempt {attempt+1} failed: {str(e)}. Retrying in {wait_time} seconds..."
        print(error_msg)  # Console logging
        st.error(error_msg)  # UI feedback
        time.sleep(5)

# LLM setup
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Prompt Templates
qa_prompt = PromptTemplate.from_template("""
You are a financial assistant. Below is the context, including intermediate steps (query variants, retrieved documents, summaries) from web sources, a company database, and Alpha Vantage API. Answer the user's question clearly and concisely, citing sources (Company DB, Web, Alpha Vantage) where relevant. For news queries, prioritize recent events (e.g., 2025). For analysis queries, include key financial metrics (e.g., P/E, EPS, revenue) if available. If data is limited, provide a general response.

**Formatting Instructions**: Ensure all words (e.g., PERatio, EPS, TTM) are written without extra spaces or new line characters between letters. For example, write "104.8 billion" not "104.8 b\ni\nl\nl\ni\no\nn".

Intermediate Steps:
- Query Variants: {query_variants}
- Retrieved Documents: {doc_summaries}

Context:
{context}

Question: {question}
Answer:
""")
# Helper Functions
def clean_query(query: str) -> str:
    query = re.sub(r'[^\w\s\-\.]', '', query)
    query = " ".join(query.split()[:20])
    return query.strip()

@st.cache_resource
def build_ticker_map(_vectorstore) -> Dict[str, str]:
    ticker_map = {}
    docs = vectorstore.get()
    for metadata in docs.get("metadatas", []):
        if metadata.get("ticker") and metadata.get("company_name"):
            ticker_map[metadata["company_name"].lower()] = metadata["ticker"]
    return ticker_map

def extract_ticker(query: str, vectorstore, ticker_map: Dict[str, str]) -> str:
    query_lower = query.lower()
    for company, ticker in ticker_map.items():
        if company in query_lower or ticker.lower() in query_lower:
            return ticker
    search_results = vectorstore.similarity_search(query, k=1)
    if search_results and "ticker" in search_results[0].metadata:
        return search_results[0].metadata["ticker"]
    prompt = f"Extract the stock ticker from this query, return only the ticker symbol or 'Unknown' if unclear:\n\"{query}\""
    response = llm.invoke(prompt).content.strip()
    normalized = response.lower()

    if normalized in ["unknown", "could not find", "none", "", "n/a", "na", "not available", "unavailable"]:
        return "Unknown"
    return response.upper()

extract_ticker_lambda = RunnableLambda(
    lambda inputs: extract_ticker(inputs["query"], inputs["vectorstore"], inputs["ticker_map"])
)

def translate_query(query: str, ticker: str) -> str:
    prompt = f"""
You're optimizing user queries for better retrieval in a stock research assistant.

Example:
Original: "any latest news on SMCI?"
Improved: "SMCI latest news 2025 earnings analyst ratings stock performance update on business"

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
    # Skip if ticker is not valid or the query isn't about a specific stock
    if ticker.lower() in ["unknown", "none", "", "n/a", "na", "not available", "unavailable"]:
        print(f"⚠️ Skipped Alpha Vantage fetch since no valid ticker provided ({ticker})")
        return {}
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
            time.sleep(2)
    return {}

fetch_alpha_vantage_lambda = RunnableLambda(
    lambda inputs: fetch_alpha_vantage_data(inputs["ticker"])
)


#def clean_text(text: str) -> str:
#    # Collapse characters separated by newlines
#    text = re.sub(r'(?:[a-zA-Z]\n){2,}[a-zA-Z]', lambda m: m.group(0).replace('\n', ''), text)
#    # Convert single line breaks into spaces (preserves paragraphs)
#    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
#    return text.strip()

summarize_prompt = PromptTemplate.from_template("""
Summarize the following financial document in a detailed but concise way (around 200–300 words), capturing key investor-relevant points like stock fundamentals, key news, market trends, company performance, analyst sentiment, or economic news:

{document}
Summary:
""")
def summarize_document(content: str) -> str:
    # Clean the input content
    #cleaned_content = clean_text(content)
    prompt = summarize_prompt.format(document=content)
    summary = llm.invoke(prompt).content.strip().replace("\n", " ")
    # Clean the summary output
#     cleaned_summary = clean_text(summary)
    return " ".join(summary.split()[:500])

summarize_document_lambda = RunnableLambda(summarize_document) #currently this lambda is not used in code. Only the function is used.


def retrieve_docs_with_fusion(query: str, ticker: str, av_data: Dict, k=3) -> Tuple[List[Document], List[str], str]:
    internal_docs = []
    results = vectorstore.similarity_search_with_score(query, k=2)
    for doc, score in results:
        # Clean the document content
        internal_docs.append((doc, score ))

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
                        content = " ".join(result["content"].split()[:1000])
                        
                        summarized = summarize_document(content)
                        # Clean the summary
#                         cleaned_summary = clean_text(summarized)
                        doc = Document(page_content=summarized, metadata={"source": result["url"]})
                        doc_emb = np.array(embedding_model.embed_documents([summarized])[0])
                        distance = np.linalg.norm(query_emb - doc_emb)
                        web_docs.append((doc, distance))
                break
            except Exception as e:
                print(f"❌ Tavily API call failed for query '{q}' (attempt {attempt+1}): {e}")     
                time.sleep(1)

    all_docs = internal_docs + web_docs
    if av_data:
        av_content = f"Alpha Vantage Data for {ticker}: Price: {av_data['price']}, P/E: {av_data['pe_ratio']}, EPS: {av_data['eps']}, Revenue: {av_data['revenue']}, Market Cap: {av_data['market_cap']}"
        #cleaned_av_content = clean_text(av_content)
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
        # Clean the final output thoroughly
        #result["result"] = clean_text(result["result"])
        #result["doc_summaries"] = clean_text(result["doc_summaries"])
        # Log to LangSmith with retries and full error handling
        try:
            for attempt in range(2):  # Retry up to 3 times
                try:
                    ls_client.create_examples(
                        dataset_id=self.dataset_id,
                        inputs=[{"query": query}],
                        outputs=[{"answer": result["result"]}]
                    )
                    print(f"Debug: Successfully logged to LangSmith for query: {query}")
                    break
                except Exception as e:
                    print(f"Debug: Failed to log to LangSmith (attempt {attempt+1}/3): {str(e)}")
                    if attempt == 1:  # Last attempt
                        print("Debug: Giving up on LangSmith logging after 2 attempts")
                    time.sleep(2)  # Wait 2 seconds before retrying
        except Exception as e:
            print(f"Debug: LangSmith logging failed entirely: {str(e)}. Proceeding without logging.")
        return result
    
# Cache the pipeline
@st.cache_resource
def get_pipeline():
    return FusionRAG(llm=llm, vectorstore=vectorstore)

# Streamlit UI
st.title("StockRAG: Financial Analysis Assistant")
st.markdown("""
Enter a financial query to get detailed insights from company profiles, web news, and financial metrics.
""")

# This input box triggers on Enter automatically
query = st.text_input("Enter your query:", placeholder="e.g., Latest news about SMCI earnings")

# This button can also trigger submission
submit = st.button("Submit")

# Run the pipeline if Enter was pressed (query has content) or Submit was clicked
if (submit or query.strip()) and query:
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
