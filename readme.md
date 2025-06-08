# StockRAG: Financial Analysis Assistant

## Overview

StockRAG is a Retrieval-Augmented Generation (RAG) system designed for financial analysis and news retrieval. It integrates a Chroma vector database (downloaded from HF Hub, \~4,000 company tickers), web search (Tavily), financial data APIs (Alpha Vantage), and a large language model (LLaMA3-70B via Groq) to deliver sourced answers to user queries, such as "What is Tesla's P/E ratio?" The system uses LangSmith for tracing pipeline steps (e.g., ticker extraction, query variants, document retrieval), making it great for demonstrating RAG in financial applications.

Key features include:

- **Ticker Extraction**: Identifies company tickers (e.g., SMCI for Super Micro Computer) using vector search and metadata.
- **Multi-Source Retrieval**: Pulls from company profiles (Chroma vector DB), web news (Tavily), and financial metrics (Alpha Vantage).
- **Query Processing**: Generates query variants for better retrieval and summarizes documents for LLM context.
- **LangSmith Tracing**: Logs pipeline steps (e.g., query rewriting, document summaries) with retry logic for reliability.
- **Structured Answers**: Provides answers with news, financial metrics (e.g., P/E, EPS), and market sentiment, citing sources.
- **Robustness**: Includes retry logic for vectorstore loading and API calls (Tavily, Alpha Vantage).

This project showcases RAG systems, focusing on modularity and real-time financial insights.

### Application Demo: 

![Demo Query](application_demo/app_screenshot1.png)

## Academic Value

StockRAG demonstrates advanced RAG techniques for financial analysis, suitable for AI, NLP, or finance projects. It highlights:

- **RAG Architecture**: Combines retrieval (vector DB, web, APIs) with generation (LLM) for context-aware answers.
- **Vector Database Utility**: Uses Chroma with \~4,000 ticker embeddings for company lookup.
- **Multi-Source Integration**: Fuses structured (Alpha Vantage) and unstructured (Tavily, vector DB) data.
- **Tracing and Debugging**: Leverages LangSmith to log pipeline steps, aiding analysis.
- **Real-World Application**: Handles real-time financial queries, showing AI’s role in decision-making.

## Installation

### Prerequisites

- Python 3.11
- API Keys:
  - Tavily for web search
  - Alpha Vantage for financial data
  - Groq for LLM access
  - LangSmith for tracing
- Local embedding model (`all-MiniLM-L6-v2`) in `local_model/` directory

### Setup
#Demo: https://stock-assistant-umangg95.streamlit.app/
1. **Clone the Repository**:

   ```bash
   git clone https://github.com/gargumang411/stockrag.git
   cd stockrag
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Requirements include:

   - `streamlit==1.37.0`
   - `langchain==0.3.1`, `langchain-community==0.3.1`, `langchain-chroma==0.1.2`, `langchain-groq==0.1.9`
   - `langsmith==0.1.99`, `alpha-vantage==2.3.1`, `sentence-transformers==3.0.1`, `numpy==1.26.4`
   - `requests==2.32.3`, `python-dotenv==1.0.1`, `chromadb==0.4.22`, `huggingface_hub==0.24.5`, `groq==0.22.0`, `pysqlite3-binary==0.5.2`

3. **Set Environment Variables**: Create a `.env` file in the project root:

   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=stock-analysis-rag-project
   TAVILY_API_KEY=your_tavily_api_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Set Up Embedding Model**:

   - Download the `all-MiniLM-L6-v2` model from Hugging Face and place it in `local_model/`.
   - The app uses this for embeddings instead of fetching online.

5. **Vector Database**:

   - The app downloads `company_vectors` from Hugging Face (`gargumang411/company_vectors`) into `./company_vectors_cache` with retry logic (3 attempts). It's created using information about ~4000 tickers imported using yahoo finance API and converted to vector embeddings using Sentence-BERT (SBERT) transformer model. 

## Usage

- Run the app via Streamlit:

  ```bash
  streamlit run app.py
  ```
- Open `http://localhost:8501`, enter a query (e.g., "Latest news about SMCI earnings"), and view the results.

## Project Structure

- `app.py`: Main RAG pipeline with ticker extraction, retrieval, and answer generation.
- `company_vectors_cache/`: Cached directory for the Chroma vector database.
- `local_model/`: Directory for the local embedding model (`all-MiniLM-L6-v2`).
- `.env`: Environment variables for API keys and LangSmith config.
- `requirements.txt`: Python dependencies.

## LangSmith Tracing

StockRAG uses LangSmith to trace pipeline steps with retry logic:

- **Ticker Extraction**: Logs identified tickers (e.g., SMCI).
- **Query Variants**: Records rewritten queries (e.g., "SMCI latest news 2025").
- **Document Retrieval**: Tracks documents from vector DB, Tavily, and Alpha Vantage.
- **Summaries and Answers**: Logs summaries and final responses. View traces in the LangSmith UI under the "stock-analysis-rag-project" project.

## Performance Notes

- **Query Time**: \~30 to 50 seconds per query due to API calls and LLM processing.
- **Optimization Opportunities**:
  - Cache Tavily and Alpha Vantage results for faster queries.
  - Batch LLM calls for query rewriting and variants.
- **Scalability**: Supports \~4,000 tickers; can be extended with more metadata.

## Potential Improvements

- **Evaluation**:
    - Build a dataset of 50+ queries for precision/recall metrics.
    - Another open-source LLM (for e.g. Llama) could be used to evaluate the results. 
- Add financial charts or visualizations.
- Enhance error handling for API failures.
- Add agentic capabilities for multi-step queries.
- Implement a user feedback loop for adaptive learning.
- Set up a cron job to keep the vector DB updated.

## Acknowledgments

- Built with LangChain, LangSmith, Tavily, Alpha Vantage, and Groq.

---

*For issues or questions, contact \[gargumang411@gmail.com\] or open an issue on GitHub.*
