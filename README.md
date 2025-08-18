# Data Analyst Agent

This project is a sophisticated Data Analyst Agent powered by AI. It can interact with various data sources, perform complex analysis, and provide insightful answers to user queries. The agent is designed to be a powerful tool for anyone looking to extract value from their data, whether it's stored in databases, PDFs, or on the web.

## Features

- **Multi-Source Data Analysis:** The agent can connect to SQL databases, analyze data from PDFs, and scrape information from websites.
- **Natural Language Interaction:** You can ask questions in plain English, and the agent will understand the intent, perform the necessary analysis, and provide answers in a human-readable format.
- **Data Visualization:** The agent can generate charts and graphs to visualize data, making it easier to understand trends and patterns.
- **Web Scraping:** It can scrape data from web pages, process it, and use it for analysis.
- **PDF Parsing:** The agent can extract text and tables from PDF documents for analysis.
- **Semantic Search:** It uses vector embeddings and semantic chunking for intelligent searching through documents and web content.
- **Pandas DataFrame Integration:** For tabular data, the agent leverages the power of pandas for data manipulation and analysis.

## Implementation Highlights

- **CrewAI Framework:** The agent is built using the CrewAI framework, which allows for the creation of autonomous AI agents.
- **FastAPI:** The agent is exposed via a REST API built with FastAPI, making it easy to integrate with other applications.
- **ONNX for Model Optimization:** We use ONNX (Open Neural Network Exchange) to optimize and run machine learning models efficiently. This is used for tasks like semantic chunking.
- **LangChain:** We use LangChain for building complex language model workflows, including agent creation and tool integration.
- **Vector Embeddings with Pinecone:** For efficient semantic search, we use Pinecone as a vector store for embeddings.

## How to Run the Project

Follow these steps to set up and run the Data Analyst Agent locally.

### 1. Set up the ONNX Model

First, you need to set up the environment for converting and running the ONNX model.

```bash
# Create and activate a virtual environment for ONNX
python -m venv venv-converter
source venv-converter/bin/activate

# Install ONNX requirements
pip install -r onnx-requirements.txt

# Run the conversion script
python convert_onnx.py

# Deactivate the ONNX environment
deactivate
```

### 2. Set up the Main Application

Now, set up the environment for the main Data Analyst Agent.

```bash
# Create and activate a new virtual environment for the main application
python -m venv .venv
source .venv/bin/activate

# Install the main application requirements
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root of the project and add the necessary environment variables. You will need API keys for Google Generative AI and Pinecone.

```
GEMINI_API_KEY="your_gemini_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
```

### 4. Run the Application

Finally, run the FastAPI application using Uvicorn.

```bash
uvicorn main:app --reload
```

The application will be running at `http://127.0.0.1:8000`. You can now interact with the Data Analyst Agent through its API endpoints.
