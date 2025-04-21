# stock-assistant: Adaptive Retrieval-Augmented Generation for Financial Analysis

## Overview
stock-assistant is an Adaptive Retrieval-Augmented Generation (RAG) system designed for real-time financial analysis and news retrieval. It integrates a Chroma vector database stored using HF Hub(~4,000 company tickers), web search (Tavily), financial data APIs (Alpha Vantage), and a large language model (Llama3-70B via Groq) to deliver detailed, sourced answers to user queries, such as “What does Tesla do?” The system leverages LangSmith for comprehensive tracing of intermediate steps (e.g., ticker extraction, query variants, document retrieval), making it ideal for demonstrations of adaptive RAG in financial applications.

Key features include:
- **Dynamic Ticker Extraction**: Identifies company tickers (e.g., SMCI for Super Micro Computer) using a vector database and metadata lookup.
- **Multi-Source Retrieval**: Combines company profiles (Chroma vector DB), recent web news (Tavily), and financial metrics (Alpha Vantage).
- **Adaptive Query Processing**: Generates query variants for enhanced retrieval and summarizes documents to fit LLM context limits.
- **LangSmith Tracing**: Logs all pipeline steps (e.g., query rewriting, document summaries) for transparency and debugging.
- **Structured Answers**: Delivers detailed responses with news, financial metrics (e.g., P/E, EPS), and market sentiment, citing sources.

This project was developed as a showcase of RAG systems, emphasizing modularity, robustness, and real-time financial insights.

## Academic Value
stock-assistant serves as a practical demonstration of advanced RAG techniques for financial analysis, suitable for projects in AI, NLP, or finance. It highlights:
- **RAG Architecture**: Combines retrieval (vector DB, web, APIs) with generation (LLM) for context-aware answers.
- **Vector Database Utility**: Uses Chroma with ~4,000 ticker embeddings for efficient company identification.
- **Multi-Source Integration**: Demonstrates fusion of structured (Alpha Vantage) and unstructured (Tavily, vector DB) data.
- **Tracing and Evaluation**: Leverages LangSmith to visualize pipeline steps, aiding debugging and performance analysis.
- **Real-World Application**: Addresses real-time financial queries, showcasing AI’s potential in decision-making.

## Installation

### Prerequisites
- Python 3.8+
- API Keys:
  - [Tavily](https://app.tavily.com/) for web search
  - [Alpha Vantage](https://www.alphavantage.co/) for financial data
  - [Groq](https://console.groq.com/) for LLM access
  - [LangSmith](https://smith.langchain.com/) for tracing
- Chroma vector database (created from scratch using yfinance web scraping) with pre-loaded company profiles (~4,000 tickers)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gargumang411/stock-assistant.git
   cd stock-assistant
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Requirements include:
   - `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-groq`, `langchain-chroma`
   - `langsmith`, `alpha_vantage`, `sentence-transformers`, `numpy`
   - `requests`, `python-dotenv`

3. **Set Environment Variables**:
   Create a `.env` file in the project root:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=stock-analysis-rag-project
   TAVILY_API_KEY=your_tavily_api_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Prepare Vector Database**:
   - Ensure a Chroma database (`company_vectors`) with company profiles and metadata (ticker, company_name) is available.
   - Example setup:
     ```python
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
     
        
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="company_vectors", embedding_function=embedding_model)
     ```

## Usage
Run the stock-assistant system to query financial news or metrics:

```python
from adaptive_rag_build import FusionRAG
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize components
llm = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key="your_groq_api_key")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="company_vectors", embedding_function=embedding_model)
qa_chain = FusionRAG(llm=llm, vectorstore=vectorstore)

# Run query
query = "any latest news on smci?"
response = qa_chain.invoke({"query": query})
print(response["result"])
```

**Example Output**:
```
**Answer**:
Super Micro Computer, Inc. (SMCI) has seen significant developments in April 2025. An independent auditing committee cleared SMCI of accounting malpractice allegations, with a new globally recognized auditor appointed, boosting investor confidence [Web: Yahoo Finance]. SMCI targets $40B in revenue for FY2026, driven by NVIDIA AI server systems, including Blackwell GPU servers [Web: Yahoo Finance]. The stock traded at ~$33.15, with a P/E of 14.90, EPS of ~$2.30, and market cap of ~$20B [Alpha Vantage]. Analyst sentiment is mixed, with Rosenblatt’s Buy rating ($60 target) and Goldman Sachs’ Sell downgrade [Web: Yahoo Finance]. [Sources: Yahoo Finance, Alpha Vantage, Company DB]
```

## Project Structure
- `adaptive_rag_build.py`: Core RAG pipeline with ticker extraction, retrieval, and answer generation.
- `company_vectors/`: Chroma vector database directory with company profiles.
- `VectorDB_Data_loader.ipynb`: To re-create or refresh the vectorDB.
- `.env`: Environment variables for API keys and LangSmith configuration.
- `requirements.txt`: Python dependencies.

## LangSmith Tracing
stock-assistant integrates LangSmith to trace all pipeline steps:
- **Ticker Extraction**: Logs identified tickers (e.g., SMCI).
- **Query Variants**: Records rewritten queries (e.g., “SMCI latest news 2025”).
- **Document Retrieval**: Tracks retrieved documents from vector DB, Tavily, and Alpha Vantage.
- **Summaries and Answers**: Logs document summaries and final responses.

View traces in the LangSmith UI under the “stock-analysis” project.

## Performance Notes
- **Query Time**: ~30 to 50 seconds per query due to multiple API calls and LLM invocations.
- **Optimization Opportunities**:
  - Cache Tavily and Alpha Vantage results to reduce latency.
  - Batch LLM calls for query rewriting and variant generation.
  - Optimize vector DB searches with precomputed ticker maps.
- **Scalability**: Handles ~4,000 tickers; extendable to more with additional metadata.

## Potential Improvements
- **Evaluation**: Create a dataset of 50+ queries for precision/recall metrics.
- **Better routing** for information retrieval for different question types.
- Integrate **financial charts or visualizations**.
- **Robustness**: Add comprehensive error handling for API failures.
- Add **agentic capabilities** for multi-step queries.
- Implement **user feedback loop** for adaptive learning.
- **Cron job** to keep the vector DB updated.


## Acknowledgments
- Built with [LangChain](https://www.langchain.com/), [LangSmith](https://smith.langchain.com/), [Tavily](https://tavily.com/), [Alpha Vantage](https://www.alphavantage.co/), and [Groq](https://groq.com/).

---

*For issues or questions, contact [gargumang411@gmail.com] or open an issue on GitHub.*
