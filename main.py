from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import re
import asyncio
from scraper_agent import run_scraping_task
from analysis_agent import run_analysis_task
from pdf_parser import run_pdf_analysis_task
import uuid
import subprocess
import tempfile
import sys
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI(title="Data Analyst AI Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

class AnalysisRequest(BaseModel):
    query: str
    data: Optional[str] = None  # For text input

# 2. Update the API response model to include a 'reasoning' field
class AgentResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    agent_used: Optional[str] = None
    reasoning: Optional[List[str]] = None

# Available agents/tools
AVAILABLE_AGENTS = {
    "scraper": {
        "name": "Web Scraper",
        "description": "Scrapes websites and extracts information based on user prompts",
        "capabilities": ["web scraping", "data extraction", "content analysis"],
        "function": run_scraping_task
    },
    "analysis": {
        "name": "Data Analyst",
        "description": "Analyzes data and provides insights",
        "capabilities": ["data analysis", "pandas", "statistics"],
        "function": run_analysis_task
    },
    "pdf_analyzer": {
        "name": "PDF Analyzer",
        "description": "Analyzes PDF files and provides insights",
        "capabilities": ["pdf analysis", "text extraction", "content analysis"],
        "function": run_pdf_analysis_task
    }
}

def create_agent_selection_prompt(query: str, data: str = None) -> str:
    """
    Creates a structured prompt for Gemini to select the appropriate agent.
    """
    prompt = f"""
You are an AI agent coordinator. Your job is to analyze the user's request and determine which agent should handle it.

AVAILABLE AGENTS:
{json.dumps(AVAILABLE_AGENTS, indent=2, default=str)}

USER REQUEST: {query}

USER DATA: {data if data else "No data provided"}

INSTRUCTIONS:
1. Analyze the user's request and available data/file content
2. Determine which agent is most appropriate
3. If no agent is suitable, respond with "no_agent"
4. If an agent is suitable, provide the agent name and any specific parameters needed
5. For scraper agent: extract URL from the request or data if mentioned
6. For file analysis: if data contains PDF, use the scrapper agent to extract the text, if data contains CSV, JSON, or other structured data, note this for future agents

RESPONSE FORMAT (JSON only):
{{
    "agent": "agent_name or no_agent",
    "reason": "explanation of why this agent was chosen",
    "parameters": {{
        "url": "if scraper agent and URL is mentioned",
        "prompt": "specific prompt for the agent"
    }}
}}

Respond with ONLY the JSON, no additional text.
"""
    return prompt

def parse_agent_response(response_text: str) -> Dict[str, Any]:
    """
    Parses the agent selection response from Gemini.
    """
    try:
        # Clean the response to extract JSON
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        return json.loads(response_text)
    except Exception as e:
        print(f"Error parsing agent response: {e}")
        return {"agent": "no_agent", "reason": "Failed to parse response"}

class WebsiteAnalysisInput(BaseModel):
    url: str = Field(description="The URL to scrape for analysis")

class WebsiteAnalysisTool(BaseTool):
    name: str = "Website Analysis Tool"
    description: str = (
        "A comprehensive tool that analyzes website content. "
        "Use this for any questions that require accessing a URL. "
        "It automatically scrapes the content, finds relevant data tables, "
        "and can perform complex data analysis, including creating charts, "
        "graphs, and visualizations to answer the user's question."
    )
    args_schema: type[BaseModel] = WebsiteAnalysisInput

    async def _run(self, **kwargs) -> str:
        """Perform data analysis."""
        try:
            # Extract task from kwargs (CrewAI passes it as a dictionary)
            url = kwargs.get('url', '')
            if isinstance(url, dict):
                url = url.get('url', '')
            
            print(f"📊 WebsiteAnalysisTool processing: {url}")
            
            # For now, this is a placeholder for general analysis
            # In the future, this could handle various data analysis tasks
            return f"Website analysis completed for URL: {url}"
            
        except Exception as e:
            print(f"❌ Error in WebsiteAnalysisTool: {str(e)}")
            return f"Error analyzing website: {str(e)}"

class ScrapingInput(BaseModel):
    url: str = Field(description="The URL of the website to scrape")
    query: str = Field(description="The user's original question or topic of interest")

class ScrapingTool(BaseTool):
    name: str = "Scraping Tool"
    description: str = (
        "Use this tool to scrape a website and prepare its content for analysis. "
        "It scrapes the text and tables from the URL and saves them for other tools to use. "
        "It returns a list of table IDs that can be used by the Dataframe Analysis Tool."
    )
    args_schema: type[BaseModel] = ScrapingInput

    async def _run(self, url: str, query: str) -> str:
        """Scrape website content and prepare it for analysis."""
        try:
            from scraper_agent import run_scraping_only_task
            result = await run_scraping_only_task(url, query)
            
            if "error" in result:
                return f"Error scraping website: {result['error']}"
            
            return json.dumps(result)
        except Exception as e:
            return f"Error during scraping: {str(e)}"

class DataframeAnalysisInput(BaseModel):
    table_id: str = Field(description="The ID of the table to analyze")
    query: str = Field(description="The analytical question to answer")

class DataframeAnalysisTool(BaseTool):
    name: str = "Dataframe Analysis Tool"
    description: str = "Analyzes tables by generating and executing Python pandas code to answer specific questions. Can also create charts and visualizations when requested."
    args_schema: type[BaseModel] = DataframeAnalysisInput

    async def _run(self, **kwargs) -> str:
        """
        Analyzes a DataFrame using LangChain's powerful Pandas Agent.
        """
        try:
            # Extract parameters from kwargs
            table_id = kwargs.get('table_id')
            query = kwargs.get('query')
            
            print(f"🔍 LangChain Agent: Analyzing table '{table_id}' with query: '{query}'")
            
            table_path = f"temp_files/{table_id}"
            if not os.path.exists(table_path):
                return json.dumps({"status": "error", "message": f"Table file not found: {table_path}"})
            
            df = pd.read_csv(table_path)

            # Initialize the LangChain LLM *inside the tool* to ensure it's the correct object type
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            llm_for_langchain = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.0)

            # Create and run the Pandas DataFrame Agent
            pandas_agent = create_pandas_dataframe_agent(
                llm_for_langchain, 
                df, 
                agent_executor_kwargs={"handle_parsing_errors": True},
                verbose=True,
                allow_dangerous_code=True, # Opt-in to code execution
                max_iterations=5,
                prefix="""You are a data analysis expert. When creating charts:
1. ALWAYS use matplotlib.use('Agg') before importing matplotlib.pyplot
2. NEVER use plt.show() - instead save charts with plt.savefig('temp_files/chart.png')
3. Use plt.close() after saving to free memory
4. Focus on answering the user's question with data-driven insights"""
            )
            
            # Use await for the async invocation
            response = await pandas_agent.ainvoke(query)
            
            answer = response.get("output", "Analysis completed, but no specific output was generated.")
            
            # The agent might save a chart. Let's check for it.
            chart_path = "temp_files/chart.png"
            if os.path.exists(chart_path):
                answer += f"\n\nA chart has been generated and saved at: {chart_path}"
            
            return json.dumps({"status": "success", "answer": answer})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return json.dumps({"status": "error", "message": f"LangChain agent failed: {str(e)}"})

# The old DataframeAnalysisTool and all its helper methods are now removed.
# LangChain's agent will handle this logic.

class PDFAnalysisInput(BaseModel):
    pdf_path: str = Field(description="The path to the PDF file to analyze")
    query: str = Field(description="The user's question about the PDF content")

class PDFAnalysisTool(BaseTool):
    name: str = "PDF Analysis Tool"
    description: str = "Analyzes PDF documents by extracting text, creating embeddings, and performing semantic search to answer questions about the content."
    args_schema: type[BaseModel] = PDFAnalysisInput

    async def _run(self, **kwargs) -> str:
        """Analyze PDF content using the full pipeline."""
        try:
            # Extract parameters from kwargs
            pdf_path = kwargs.get('pdf_path')
            query = kwargs.get('query')
            
            print(f"📄 PDFAnalysisTool: Analyzing PDF '{pdf_path}' with query: '{query}'")
            
            # Call the existing PDF analysis function
            from pdf_parser import run_pdf_analysis_task
            result = await run_pdf_analysis_task(pdf_path, query)
            
            if "error" in result:
                return json.dumps({"status": "error", "message": result["error"]})
            
            # Extract the answer from the search analysis
            search_analysis = result.get("search_analysis", {})
            
            # Handle the new response structure from search_and_analyze
            if isinstance(search_analysis, dict) and "analysis" in search_analysis:
                # New structure: search_analysis.analysis.answer
                answer = search_analysis.get("analysis", {}).get("answer", "No answer found in the PDF.")
                confidence = search_analysis.get("analysis", {}).get("confidence", "unknown")
                method = search_analysis.get("method", "unknown")
                
                return json.dumps({
                    "status": "success", 
                    "answer": answer,
                    "confidence": confidence,
                    "method": method
                })
            elif isinstance(search_analysis, dict) and "answer" in search_analysis:
                # Legacy structure: search_analysis.answer
                answer = search_analysis.get("answer", "No answer found in the PDF.")
                return json.dumps({"status": "success", "answer": answer})
            else:
                # Fallback: return the raw search_analysis
                return json.dumps({"status": "success", "answer": str(search_analysis)})
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return json.dumps({"status": "error", "message": f"PDF analysis failed: {str(e)}"})

# Simple reasoning logger that captures agent output
class ReasoningLogger:
    """
    Simple logger to capture agent reasoning steps from the output.
    """
    def __init__(self):
        self.reasoning_steps: List[str] = []

    def add_step(self, step: str):
        """Add a reasoning step."""
        if step.strip():
            step_number = len(self.reasoning_steps) + 1
            self.reasoning_steps.append(f"Step {step_number}: {step}")

async def create_crewai_workflow(user_query: str, file_path: Optional[str] = None):
    """
    Creates and executes a CrewAI workflow with a single, powerful Universal Data Analyst agent.
    """
    # Configure the Gemini API key before initializing any components that need it
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize the LLM for CrewAI using the required string format
    llm_for_crewai = "gemini/gemini-1.5-flash"

    # Initialize the reasoning logger
    reasoning_logger = ReasoningLogger()

    # The old specialist agents have been removed.
    # The new Universal Data Analyst agent handles all tasks.
    universal_analyst = Agent(
        role='Universal Data Analyst',
        goal='Answer the user\'s query by analyzing the provided data from URLs, files, or text. Prioritize using internal knowledge and reasoning first.',
        backstory=(
            "You are a powerful AI assistant with advanced reasoning and web access capabilities. "
            "Your primary approach is to directly answer the user's question using your own knowledge and abilities. "
            "You must follow a strict reasoning process:\n"
            "1. First, attempt to solve the user's query on your own without using any tools.\n"
            "2. If you fail, you MUST explain in your thought process exactly WHY you failed (e.g., 'My direct attempt to access the URL failed because the content is dynamic and I cannot parse the tables correctly').\n"
            "3. Only after articulating your failure may you use one of the following specialized tools:\n\n"
            "- Use the Scraping Tool as a fallback if you fail to access or parse a URL directly.\n"
            "- Use the PDF Analysis Tool as a fallback if you fail to access or parse a PDF or Docx file directly.\n"
            "- Use the Dataframe Analysis Tool when you have tabular data (like from a CSV) and the query requires specific numerical calculations, statistical analysis, or visualizations that you cannot perform natively."
        ),
        tools=[ScrapingTool(), DataframeAnalysisTool(), PDFAnalysisTool()],
        llm=llm_for_crewai,  # Use the string-based LLM for CrewAI
        verbose=True
    )

    # Create a single, comprehensive task for the agent
    task_description = f"Answer the following query: '{user_query}'"
    if file_path:
        task_description += f"\n\nAnalyze the content of the file located at: {file_path}"
    
    analysis_task = Task(
        description=task_description,
        expected_output="A comprehensive answer that directly addresses the user's question, supported by data and analysis.",
        agent=universal_analyst
    )

    # Create and execute the crew
    crew = Crew(
        agents=[universal_analyst],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True
    )

    try:
        print("🚀 Executing the Universal Data Analyst workflow...")
        
        # Execute the crew and capture the result
        result = await crew.kickoff_async()
        
        # For now, we'll capture basic reasoning from the verbose output
        # In a future version, we can implement proper event handling
        reasoning_logger.add_step("Agent executed successfully")
        
        print("✅ Workflow completed successfully.")
        return result, reasoning_logger.reasoning_steps
    except Exception as e:
        print(f"❌ Crew workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"An unexpected error occurred in the workflow: {str(e)}"}, []

@app.get("/")
async def root():
    return {"message": "Data Analyst AI Agent API", "version": "1.0.0"}

@app.get("/agents")
async def get_agents():
    """Get list of available agents"""
    return {
        "agents": AVAILABLE_AGENTS,
        "total": len(AVAILABLE_AGENTS)
    }

@app.post("/analyze", response_model=AgentResponse)
async def analyze_data(query: str = Form(...), file: UploadFile = File(None)):
    """
    This endpoint takes a user query, handles an optional file upload, 
    and runs the hierarchical agent workflow.
    """
    temp_file_path = None
    try:
        # Clean up any existing temp files before starting
        cleanup_temp_files()
        
        final_query = query
        if file:
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            # Use a secure way to create a temporary file path
            temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
            
            with open(temp_file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # The query to the agent should not contain the local file path for security
            final_query += f" (File Name: {file.filename})"

        # Execute the workflow and get result and reasoning
        result, reasoning_steps = await create_crewai_workflow(final_query, file_path=temp_file_path)
        
        # Parse the final result correctly
        if hasattr(result, 'raw'):
            final_answer = result.raw
        elif hasattr(result, 'output'):
            final_answer = result.output
        elif isinstance(result, str):
            final_answer = result
        elif isinstance(result, dict) and "error" in result:
            final_answer = f"Error: {result['error']}"
        else:
            final_answer = str(result)
        
        return AgentResponse(
            success=True,
            result={"answer": final_answer},
            message="Successfully executed CrewAI workflow.",
            agent_used="crewai",
            reasoning=reasoning_steps
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return AgentResponse(
            success=False,
            result={"error": str(e)},
            message=f"An unexpected error occurred: {str(e)}",
            agent_used="none"
        )
    finally:
        # Clean up the uploaded file after the request is complete
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def cleanup_temp_files():
    """
    Clean up all temporary files before starting a new analysis task.
    """
    try:
        if os.path.exists("temp_files"):
            # Remove all CSV and temporary files, but keep uploaded PDFs temporarily
            for filename in os.listdir("temp_files"):
                if filename.endswith('.csv') or filename.startswith('table_') or filename.startswith('pdf_table_'):
                    file_path = os.path.join("temp_files", filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"⚠️ Error deleting {file_path}: {e}")
            print(f"🧹 Cleaned up old table files from temp_files")
        else:
            # Create the directory if it doesn't exist
            os.makedirs("temp_files", exist_ok=True)
            print(f"📁 Created temp_files directory")
    except Exception as e:
        print(f"❌ Error during cleanup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

