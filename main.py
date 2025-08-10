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
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import sqlalchemy
from sqlalchemy import create_engine
import concurrent.futures
from dataclasses import dataclass
from enum import Enum

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

# New Planner-Executor Architecture Models
class TaskDependency(BaseModel):
    task_id: str
    dependencies: List[str] = []

class ExecutionTask(BaseModel):
    task_id: str
    tool: str
    description: str
    dependencies: List[str] = []
    output_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ExecutionStage(BaseModel):
    stage_id: int
    description: str
    tasks: List[ExecutionTask]
    can_run_parallel: bool = True

class ExecutionPlan(BaseModel):
    context: Dict[str, Any]
    stages: List[ExecutionStage]
    metadata: Dict[str, Any] = {}

class DataCache:
    """In-memory cache for storing intermediate data between tasks"""
    
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()
    
    async def store(self, key: str, data: Any):
        async with self._lock:
            self._cache[key] = data
    
    async def retrieve(self, key: str) -> Any:
        async with self._lock:
            return self._cache.get(key)
    
    async def clear(self):
        async with self._lock:
            self._cache.clear()
    
    def get_all_keys(self) -> List[str]:
        return list(self._cache.keys())

class PlannerAgent:
    """Agent responsible for analyzing complex queries and creating execution plans"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def create_execution_plan(self, user_query: str, file_path: Optional[str] = None) -> ExecutionPlan:
        """
        Analyzes the user query and creates a structured execution plan (DAG)
        """
        # Create the planning prompt
        planning_prompt = self._create_planning_prompt(user_query, file_path)
        
        try:
            print(f"🤖 Planning prompt created, length: {len(planning_prompt)}")
            
            # Get the LLM response for planning
            response = await self._get_llm_response(planning_prompt)
            print(f"🤖 LLM response received, length: {len(response)}")
            print(f"🤖 LLM response preview: {response[:200]}...")
            
            # Parse the response into an execution plan
            execution_plan = self._parse_planning_response(response)
            print(f"✅ Execution plan created successfully with {len(execution_plan.stages)} stages")
            
            return execution_plan
            
        except Exception as e:
            print(f"❌ Planning failed: {str(e)}")
            raise
    
    def _create_planning_prompt(self, user_query: str, file_path: Optional[str] = None) -> str:
        """Creates a comprehensive prompt for the LLM to generate an execution plan"""
        
        prompt = f"""
You are an expert AI task planner. Your job is to analyze a user's request and create a structured execution plan that can be executed efficiently.

USER REQUEST:
{user_query}

AVAILABLE TOOLS:
- ScrapingTool: For web scraping and data extraction
- DataframeAnalysisTool: For running pandas code to answer questions on csv files and other scraped structured data
- PDFAnalysisTool: For PDF document scraping and data extraction
- SQLAnalysisTool: For SQL database analysis (DuckDB, PostgreSQL, MySQL, etc.)
- NoSQLAnalysisTool: For NoSQL database analysis (MongoDB, Redis, etc.)

PLANNING INSTRUCTIONS:
1. Analyze the user's request to identify:
   - What data sources are needed (URLs, databases, files, etc.)
   - What analytical questions need to be answered
   - Dependencies between different tasks
   - Which tools are most appropriate for each task

2. Create a staged execution plan:
   - Stage 1: Data Acquisition (must happen first, sequentially if all of the questions are from the same source)
   - Stage 2: Analysis (can run in parallel where possible)

3. For SQL Analysis tasks:
   - Use SQLAnalysisTool when the user asks about database queries
   - Include database_info in context with type and connection details
   - Pass table_info if available
                   - SQLAnalysisTool handles SQL queries and data retrieval ONLY
                - For plotting/visualization and analysis requests, create a separate task using DataframeAnalysisTool
                - SQLAnalysisTool should output data that can be used by DataframeAnalysisTool for visualization or analysis
                - IMPORTANT: If a question asks for both SQL analysis AND plotting, create TWO separate tasks:
                  1. First task: Use SQLAnalysisTool to get the data
                  2. Second task: Use DataframeAnalysisTool to create visualizations (depends on first task)

4. For each task, specify:
   - task_id: unique identifier
   - tool: which tool to use
   - description: what the task should do
   - dependencies: list of task_ids this task depends on
   - output_id: if this task produces data for other tasks to use

5. Consider task dependencies carefully:
   - If Question B depends on Question A's result, B must wait for A
   - If Questions A and B are independent, they can run in parallel
   - Data acquisition (scraping, database connection) should happen once and be shared

6. Return your response as a valid JSON object with this exact structure:
{{
  "context": {{
    "data_source_type": "web_scraping|database|file|mixed",
    "urls": ["url1", "url2"],
    "database_info": {{"type": "duckdb", "connection": "s3://..."}},
    "file_info": "description of file content"
  }},
  "stages": [
    {{
      "stage_id": 1,
      "description": "Data acquisition and preparation",
      "tasks": [
        {{
          "task_id": "unique_id",
          "tool": "tool_name",
          "description": "What this task does",
          "dependencies": [],
          "output_id": "data_reference_id"
        }}
      ]
    }},
    {{
      "stage_id": 2,
      "description": "Data analysis and visualization",
      "tasks": [
        {{
          "task_id": "unique_id",
          "tool": "tool_name",
          "description": "What this task does",
          "dependencies": ["dependency_task_id"],
          "output_id": "optional_output_id"
        }}
      ]
    }}
  ]
}}

IMPORTANT: Ensure your response is valid JSON. Do not include any explanatory text outside the JSON object.

EXAMPLES:

For SQL Analysis with plotting (when user asks about database queries and visualization):
{{
  "context": {{
    "data_source_type": "database",
    "database_info": {{"type": "duckdb", "connection": "s3://bucket/path/data.parquet?s3_region=region"}},
    "table_info": "columns: court_code, title, description, judge, pdf_link, cnr, date_of_registration, decision_date, disposal_nature, court, raw_html, bench, year"
  }},
  "stages": [
    {{
      "stage_id": 1,
      "description": "Database connection and analysis",
      "tasks": [
        {{
          "task_id": "sql_analysis",
          "tool": "SQLAnalysisTool",
          "description": "Execute SQL queries to retrieve data for analysis",
          "dependencies": [],
          "output_id": "sql_results"
        }}
      ]
    }},
    {{
      "stage_id": 2,
      "description": "Data visualization and analysis",
      "tasks": [
        {{
          "task_id": "plotting",
          "tool": "DataframeAnalysisTool",
          "description": "Create visualizations and perform analysis on the SQL results",
          "dependencies": ["sql_analysis"],
          "output_id": "visualization"
        }}
      ]
    }}
  ]
}}

For multiple SQL questions (when user asks several database questions):
{{
  "context": {{
    "data_source_type": "database",
    "database_info": {{"type": "duckdb", "connection": "s3://bucket/path/data.parquet?s3_region=region"}},
    "table_info": "columns: court_code, title, description, judge, pdf_link, cnr, date_of_registration, decision_date, disposal_nature, court, raw_html, bench, year"
  }},
  "stages": [
    {{
      "stage_id": 1,
      "description": "Database connection and analysis",
      "tasks": [
        {{
          "task_id": "question_1",
          "tool": "SQLAnalysisTool",
          "description": "Answer the first question using SQL queries",
          "dependencies": [],
          "output_id": "result_1"
        }},
        {{
          "task_id": "question_2",
          "tool": "SQLAnalysisTool",
          "description": "Answer the second question using SQL queries",
          "dependencies": [],
          "output_id": "result_2"
        }},
        {{
          "task_id": "question_3",
          "tool": "SQLAnalysisTool",
          "description": "Answer the third question using SQL queries",
          "dependencies": [],
          "output_id": "result_3"
        }}
      ]
    }}
  ]
}}
"""
        
        if file_path:
            prompt += f"\n\nFILE CONTENT: The user has uploaded a file at {file_path}. Consider this in your planning."
        
        return prompt
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Gets response from the LLM for planning"""
        try:
            # For ChatGoogleGenerativeAI, we need to use the invoke method
            if hasattr(self.llm, 'invoke'):
                response = await self.llm.ainvoke(prompt)
                return response.content
            else:
                # Fallback for other LLM types
                return str(await self.llm.agenerate([prompt]))
        except Exception as e:
            print(f"❌ LLM planning request failed: {str(e)}")
            raise
    
    def _parse_planning_response(self, response: str) -> ExecutionPlan:
        """Parses the LLM response into an ExecutionPlan object"""
        try:
            # Clean the response to extract just the JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse the JSON
            plan_data = json.loads(response)
            
            print(f"🔍 Parsed plan_data keys: {list(plan_data.keys())}")
            if 'context' in plan_data:
                print(f"🔍 Context keys: {list(plan_data['context'].keys())}")
                if 'database_info' in plan_data['context']:
                    print(f"🔍 database_info: {plan_data['context']['database_info']}")
            
            # Convert to ExecutionPlan object
            stages = []
            for stage_data in plan_data.get("stages", []):
                tasks = []
                for task_data in stage_data.get("tasks", []):
                    task = ExecutionTask(
                        task_id=task_data["task_id"],
                        tool=task_data.get("tool", ""),
                        description=task_data["description"],
                        dependencies=task_data.get("dependencies", []),
                        output_id=task_data.get("output_id"),
                        context=task_data.get("context", {})
                    )
                    tasks.append(task)
                
                stage = ExecutionStage(
                    stage_id=stage_data["stage_id"],
                    description=stage_data["description"],
                    tasks=tasks,
                    can_run_parallel=stage_data.get("can_run_parallel", True)
                )
                stages.append(stage)
            
            return ExecutionPlan(
                context=plan_data.get("context", {}),
                stages=stages,
                metadata=plan_data.get("metadata", {})
            )
            
        except Exception as e:
            print(f"❌ Failed to parse planning response: {str(e)}")
            print(f"Raw response: {response}")
            raise

class ExecutorCrew:
    """Executes the execution plan created by the Planner Agent"""
    
    def __init__(self, data_cache: DataCache, tools: Dict[str, Any]):
        self.data_cache = data_cache
        self.tools = tools
    
    async def execute_plan(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Executes the execution plan stage by stage, handling dependencies and parallel execution
        """
        print(f"🚀 Executing execution plan with {len(execution_plan.stages)} stages")
        
        # Store the execution plan context for tools to access
        self.execution_plan_context = execution_plan.context
        
        results = {}
        
        # Execute stages sequentially
        for stage in execution_plan.stages:
            print(f"📋 Executing Stage {stage.stage_id}: {stage.description}")
            
            if stage.can_run_parallel and len(stage.tasks) > 1:
                # Execute tasks in parallel where possible
                stage_results = await self._execute_stage_parallel(stage, results)
            else:
                # Execute tasks sequentially
                stage_results = await self._execute_stage_sequential(stage, results)
            
            # Store stage results
            results.update(stage_results)
            
            print(f"✅ Stage {stage.stage_id} completed")
        
        return results
    
    async def _execute_stage_parallel(self, stage: ExecutionStage, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Executes tasks in a stage in parallel, respecting dependencies"""
        
        # Group tasks by dependency level
        dependency_groups = self._group_tasks_by_dependencies(stage.tasks, previous_results)
        
        stage_results = {}
        
        # Execute each dependency group in parallel
        for group in dependency_groups:
            print(f"  🔄 Executing {len(group)} tasks in parallel")
            
            # Create tasks for parallel execution
            tasks = []
            for task in group:
                task_coro = self._execute_single_task(task, previous_results, stage_results)
                tasks.append(task_coro)
            
            # Execute all tasks in the group concurrently
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and store outputs
            for i, task in enumerate(group):
                try:
                    result = group_results[i]
                    if isinstance(result, Exception):
                        print(f"    ❌ Task {task.task_id} failed: {str(result)}")
                        stage_results[task.task_id] = {"error": str(result)}
                    else:
                        print(f"    ✅ Task {task.task_id} completed")
                        stage_results[task.task_id] = result
                        
                        # Store output in cache if specified
                        if task.output_id:
                            await self.data_cache.store(task.output_id, result)
                            
                except Exception as e:
                    print(f"    ❌ Task {task.task_id} failed: {str(e)}")
                    stage_results[task.task_id] = {"error": str(e)}
        
        return stage_results
    
    async def _execute_stage_sequential(self, stage: ExecutionStage, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Executes tasks in a stage sequentially"""
        
        stage_results = {}
        
        for task in stage.tasks:
            print(f"  📝 Executing task: {task.task_id}")
            
            try:
                result = await self._execute_single_task(task, previous_results, stage_results)
                stage_results[task.task_id] = result
                
                # Store output in cache if specified
                if task.output_id:
                    await self.data_cache.store(task.output_id, result)
                    
                print(f"    ✅ Task {task.task_id} completed")
                
            except Exception as e:
                print(f"    ❌ Task {task.task_id} failed: {str(e)}")
                stage_results[task.task_id] = {"error": str(e)}
        
        return stage_results
    
    def _group_tasks_by_dependencies(self, tasks: List[ExecutionTask], previous_results: Dict[str, Any]) -> List[List[ExecutionTask]]:
        """Groups tasks by dependency level for parallel execution"""
        
        # Create a dependency graph
        dependency_graph = {}
        for task in tasks:
            dependency_graph[task.task_id] = set(task.dependencies)
        
        # Find tasks that can run in parallel (no dependencies or dependencies met)
        groups = []
        remaining_tasks = set(task.task_id for task in tasks)
        completed_tasks = set(previous_results.keys())
        
        while remaining_tasks:
            # Find tasks that can run now (all dependencies met)
            ready_tasks = []
            for task_id in remaining_tasks:
                task = next(t for t in tasks if t.task_id == task_id)
                if all(dep in completed_tasks for dep in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Deadlock or circular dependency - run remaining tasks sequentially
                remaining_task_objects = [t for t in tasks if t.task_id in remaining_tasks]
                groups.append(remaining_task_objects)
                break
            
            # Add ready tasks to current group
            groups.append(ready_tasks)
            
            # Mark these tasks as completed for next iteration
            for task in ready_tasks:
                remaining_tasks.remove(task.task_id)
                completed_tasks.add(task.task_id)
        
        return groups
    
    async def _execute_single_task(self, task: ExecutionTask, previous_results: Dict[str, Any], current_stage_results: Dict[str, Any]) -> Any:
        """Executes a single task using the appropriate tool"""
        
        # Get the tool
        tool_name = task.tool
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        
        # Prepare context for the tool
        context = self._prepare_task_context(task, previous_results, current_stage_results)
        
        # Execute the tool
        if hasattr(tool, '_run'):
            result = await tool._run(**context)
        else:
            # Fallback for sync tools
            result = tool._run(**context)
        
        return result
    
    def _prepare_task_context(self, task: ExecutionTask, previous_results: Dict[str, Any], current_stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the context for a task execution"""
        
        context = {}
        
        # Add task-specific context
        if task.context:
            context.update(task.context)
        
        # Add results from previous stages
        context.update(previous_results)
        
        # Add results from current stage (for dependencies within the same stage)
        context.update(current_stage_results)
        
        # Add the execution plan context (contains URLs, database info, etc.)
        if hasattr(self, 'execution_plan_context'):
            context.update(self.execution_plan_context)
        
        # Add the data cache so tools can access it
        context["data_cache"] = self.data_cache
        
        # Add the task description as query if not present
        if "query" not in context:
            context["query"] = task.description
        
        return context

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

    async def _run(self, **kwargs) -> str:
        """Scrape website content and prepare it for analysis."""
        try:
            # Extract parameters from kwargs (context-aware)
            url = kwargs.get('url')
            query = kwargs.get('query')
            
            print(f"🔍 ScrapingTool debug - kwargs keys: {list(kwargs.keys())}")
            print(f"🔍 ScrapingTool debug - url from kwargs: {url}")
            
            # If no URL provided, try to extract from context
            if not url and 'urls' in kwargs:
                url = kwargs['urls'][0] if kwargs['urls'] else None
                print(f"🔍 ScrapingTool debug - found url from 'urls': {url}")
            
            if not url:
                print(f"❌ ScrapingTool: No URL found in context. Available keys: {list(kwargs.keys())}")
                return "Error: No URL provided for scraping"
            
            print(f"🌐 ScrapingTool: Scraping URL '{url}' for query: '{query}'")
            
            from scraper_agent import run_scraping_only_task
            result = await run_scraping_only_task(url, query)
            
            if "error" in result:
                return f"Error scraping website: {result['error']}"
            
            # Store the result in the data cache for other tools to use
            if 'data_cache' in kwargs:
                data_cache = kwargs['data_cache']
                # Store the scraped data with a key that other tools can find
                await data_cache.store('scraped_data', result)
                print(f"💾 ScrapingTool: Stored scraped data in cache")
            
            return json.dumps(result)
        except Exception as e:
            return f"Error during scraping: {str(e)}"

class DataframeAnalysisInput(BaseModel):
    table_id: Optional[str] = Field(description="The ID of the table to analyze (optional if using cached data)")
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
            # Extract parameters from kwargs (context-aware)
            table_id = kwargs.get('table_id')
            query = kwargs.get('query')
            
            # Debug: Print what's available in kwargs
            print(f"🔍 DataframeAnalysisTool: Available kwargs keys: {list(kwargs.keys())}")
            if 'question_2' in kwargs:
                print(f"🔍 DataframeAnalysisTool: question_2 result type: {type(kwargs['question_2'])}")
                print(f"🔍 DataframeAnalysisTool: question_2 result preview: {str(kwargs['question_2'])[:200]}...")
            
            # If no table_id provided, try to extract from context or cache
            if not table_id:
                # First try to get from context
                if 'context' in kwargs:
                    context = kwargs['context']
                if isinstance(context, dict):
                    # Look for table references in context
                    for key, value in context.items():
                        if isinstance(value, str) and value.startswith('table_'):
                            table_id = value
                            break
            
                # If still no table_id, try to get from data cache
                if not table_id and 'data_cache' in kwargs:
                    data_cache = kwargs['data_cache']
                    try:
                        scraped_data = await data_cache.retrieve('scraped_data')
                        
                        # Handle both string and dict formats
                        if isinstance(scraped_data, str):
                            try:
                                # Parse JSON string if it's a string
                                scraped_data = json.loads(scraped_data)
                            except json.JSONDecodeError:
                                scraped_data = None
                        
                        if scraped_data and isinstance(scraped_data, dict):
                            # Look for available_tables first
                            if 'available_tables' in scraped_data and scraped_data['available_tables']:
                                table_id = scraped_data['available_tables'][0]  # Use first table
                            else:
                                # Look for table files in the scraped data
                                for key, value in scraped_data.items():
                                    if isinstance(value, str) and value.startswith('table_'):
                                        table_id = value
                                        break
                    except Exception as e:
                        pass
            
            # Check if we have SQL analysis results in the cache
            sql_result_data = None
            if 'data_cache' in kwargs:
                try:
                    sql_result_data = await kwargs['data_cache'].retrieve("latest_sql_result")
                except Exception as e:
                    pass
            
            # Also check if we have data from previous stage results in context
            # Look for task results in order of preference: task_2, task_1_2, question_2
            if sql_result_data is None:
                for task_key in ['task_2', 'task_1_2', 'question_2']:
                    if task_key in kwargs:
                        task_result = kwargs[task_key]
                        print(f"🔍 DataframeAnalysisTool: Checking {task_key}: {type(task_result)}")
                        print(f"🔍 DataframeAnalysisTool: {task_key} content preview: {str(task_result)[:300]}...")
                        
                        if isinstance(task_result, str) and 'Result:' in task_result:
                            # Extract the result data from the string
                            try:
                                # Parse the result string to extract the actual data
                                import re
                                import ast
                                
                                # Look for the result pattern: "Result: [{...}, {...}]"
                                result_match = re.search(r'Result: (\[.*\])', task_result)
                                if result_match:
                                    result_str = result_match.group(1)
                                    # Convert string representation back to list of dicts
                                    sql_result_data = ast.literal_eval(result_str)
                                    print(f"🔍 DataframeAnalysisTool: Found data in {task_key} with {len(sql_result_data)} rows")
                                    break
                            except Exception as e:
                                print(f"Failed to parse {task_key} result: {e}")
                                continue
            
            if sql_result_data is not None:
                # Use the SQL result data for analysis/plotting
                # Convert list of dictionaries to pandas DataFrame
                df = pd.DataFrame(sql_result_data)
                print(f"🔍 DataframeAnalysisTool: Using SQL result data with {len(df)} rows")
                print(f"🔍 DataframeAnalysisTool: DataFrame columns: {list(df.columns)}")
                print(f"🔍 DataframeAnalysisTool: DataFrame sample:\n{df.head()}")
                
                # Don't clear the cache immediately - let it persist for this session
                # Only clear if we're done with all operations
                # if 'data_cache' in kwargs:
                #     await kwargs['data_cache'].store("latest_sql_result", None)
            elif table_id:
                # Use the table file as before
                table_path = f"temp_files/{table_id}"
                if not os.path.exists(table_path):
                    return json.dumps({"status": "error", "message": f"Table file not found: {table_path}"})
                
                df = pd.read_csv(table_path)
            else:
                return "Error: No data provided for analysis. Please ensure data is scraped first or SQL analysis is completed."

            # --- Data Validation Guardrail ---
            if df.empty:
                return json.dumps({"status": "error", "answer": "The data provided is empty. I cannot perform any analysis."})
            
            # For plotting, we need at least 2 data points
            if "plot" in query.lower() or "scatter" in query.lower() or "graph" in query.lower():
                if len(df) < 2:
                    return json.dumps({
                        "status": "error", 
                        "answer": f"I cannot create a plot because there is not enough data. The dataset only contains {len(df)} row(s). Meaningful plotting requires at least two data points."
                    })
            # ---------------------------------

            # Initialize the LangChain LLM *inside the tool* to ensure it's the correct object type
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            llm_for_langchain = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.0)

            # Create and run the Pandas DataFrame Agent
            # IMPORTANT: Disable SQL generation to prevent JULIANDAY function errors
            pandas_agent = create_pandas_dataframe_agent(
                llm_for_langchain, 
                df, 
                agent_executor_kwargs={"handle_parsing_errors": True},
                prefix="""You are a data analysis expert. You can ONLY use Python pandas operations. 
                DO NOT generate SQL queries. Use only pandas methods like:
                - df.groupby(), df.agg(), df.mean(), df.count()
                - df.plot(), df.scatter(), plt.scatter(), plt.plot()
                - df['column'].dt.days, df['column'].dt.year for date operations
                - df['column'].astype() for type conversions
                
                NEVER use SQL functions like JULIANDAY, STRFTIME, etc. Only use pandas datetime operations.
                
                Your primary directive is to use ONLY the data provided in the dataframe.
                If the data is insufficient to answer the question, you MUST respond by stating what is missing.
                DO NOT create, invent, or hallucinate sample or dummy data under any circumstances.

                CRITICAL FOR REGRESSION ANALYSIS:
                - ALWAYS start by examining the data structure: print(df.info()), print(df.head())
                - For date-based regression analysis, you MUST:
                  1. Convert date columns to datetime: df['date_column'] = pd.to_datetime(df['date_column'])
                  2. Calculate the dependent variable (e.g., delay in days): df['delay_days'] = (df['decision_date'] - df['date_of_registration']).dt.days
                  3. Use the year as the independent variable: df['year'] = df['year'].astype(int)
                  4. Import and use sklearn: from sklearn.linear_model import LinearRegression
                  5. Fit the model: model = LinearRegression(); model.fit(df[['year']], df['delay_days'])
                  6. Get the slope: slope = model.coef_[0]
                
                SPECIFIC EXAMPLE FOR COURT DATA:
                - If you have columns: date_of_registration, decision_date, year
                - Convert dates: df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], format='%d-%m-%Y')
                - Calculate delay: df['delay_days'] = (df['decision_date'] - df['date_of_registration']).dt.days
                - Use year as X: X = df[['year']].values; y = df['delay_days'].values
                - Fit regression: model = LinearRegression(); model.fit(X, y)
                - Get slope: slope = model.coef_[0]

                When charts are requested:
1. ALWAYS use matplotlib.use('Agg') before importing matplotlib.pyplot
2. NEVER use plt.show() - instead save charts with plt.savefig('temp_files/chart.png')
3. Use plt.close() after saving to free memory
                4. Focus on answering the user's question with data-driven insights from the provided dataframe.""",
                verbose=True,
                allow_dangerous_code=True, # Opt-in to code execution
                max_iterations=5
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
            # Extract parameters from kwargs (context-aware)
            pdf_path = kwargs.get('pdf_path')
            query = kwargs.get('query')
            
            # If no pdf_path provided, try to extract from context
            if not pdf_path and 'context' in kwargs:
                context = kwargs['context']
                if isinstance(context, dict) and 'file_path' in context:
                    pdf_path = context['file_path']
            
            if not pdf_path:
                return "Error: No PDF path provided for analysis"
            
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

# New Advanced Data Analysis Tools

class SQLAnalysisInput(BaseModel):
    query: str = Field(description="The analytical question to answer about the data")
    database_type: str = Field(description="Type of database: 'duckdb', 'postgresql', 'mysql', 'sqlite', etc.")
    connection_string: Optional[str] = Field(description="Database connection string or S3 path for DuckDB")
    table_info: Optional[str] = Field(description="Information about available tables/schemas")

class SQLAnalysisTool(BaseTool):
    name: str = "SQL Analysis Tool"
    description: str = (
        "Analyzes data using SQL queries. This tool connects to various SQL databases including DuckDB for S3 data, "
        "PostgreSQL, MySQL, and others. It uses LLM-generated SQL with direct database execution for robust analysis. "
        "Use this for complex analytical questions that require SQL operations like aggregations, joins, and statistical analysis. "
        "For plotting/visualization requests, this tool will prepare data for the DataframeAnalysisTool to use."
    )
    args_schema: type[BaseModel] = SQLAnalysisInput
    llm: Optional[Any] = None
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        if not self.llm:
            return "Error: LLM not available"
        
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"
    
    def _adapt_sql_for_database(self, sql_query: str, database_type: str) -> str:
        """Adapt SQL query for specific database type by replacing incompatible functions."""
        adapted_sql = sql_query
        
        if database_type.lower() == 'duckdb':
            # Replace JULIANDAY functions with DuckDB equivalents
            # Handle various spacing and case variations
            adapted_sql = adapted_sql.replace('JULIANDAY(', 'DATE_DIFF(\'day\', \'1970-01-01\', ')
            adapted_sql = adapted_sql.replace('julianday(', 'DATE_DIFF(\'day\', \'1970-01-01\', ')
            adapted_sql = adapted_sql.replace('JULIANDAY (', 'DATE_DIFF(\'day\', \'1970-01-01\', ')
            adapted_sql = adapted_sql.replace('julianday (', 'DATE_DIFF(\'day\', \'1970-01-01\', ')
            
            # Replace other common incompatible functions
            adapted_sql = adapted_sql.replace('STRFTIME(', 'STRPTIME(')
            adapted_sql = adapted_sql.replace('strftime(', 'STRPTIME(')
            adapted_sql = adapted_sql.replace('STRFTIME (', 'STRPTIME(')
            adapted_sql = adapted_sql.replace('strftime (', 'STRPTIME(')
            
            # Special case: Replace JULIANDAY(date1) - JULIANDAY(date2) pattern
            # This is a common pattern for date differences
            import re
            
            # Pattern to match JULIANDAY(date1) - JULIANDAY(date2)
            # Handle both uppercase and lowercase, including nested function calls
            patterns = [
                (r'JULIANDAY\(([^)]+)\)\s*-\s*JULIANDAY\(([^)]+)\)', r'DATE_DIFF(\'day\', \2, \1)'),
                (r'julianday\(([^)]+)\)\s*-\s*julianday\(([^)]+)\)', r'DATE_DIFF(\'day\', \2, \1)'),
                (r'JULIANDAY\s*\(([^)]+)\)\s*-\s*JULIANDAY\s*\(([^)]+)\)', r'DATE_DIFF(\'day\', \2, \1)'),
                (r'julianday\s*\(([^)]+)\)\s*-\s*julianday\s*\(([^)]+)\)', r'DATE_DIFF(\'day\', \2, \1)')
            ]
            
            for pattern, replacement in patterns:
                adapted_sql = re.sub(pattern, replacement, adapted_sql, flags=re.IGNORECASE)
            
            # Handle nested function calls like JULIANDAY(STRFTIME(...))
            # First replace STRFTIME with STRPTIME
            adapted_sql = re.sub(r'STRFTIME\(([^)]+)\)', r'STRPTIME(\1)', adapted_sql, flags=re.IGNORECASE)
            adapted_sql = re.sub(r'strftime\(([^)]+)\)', r'STRPTIME(\1)', adapted_sql, flags=re.IGNORECASE)
            
            # Then handle any remaining JULIANDAY calls that weren't caught by the subtraction pattern
            adapted_sql = re.sub(r'JULIANDAY\s*\(([^)]+)\)', r'DATE_DIFF(\'day\', \'1970-01-01\', \1)', adapted_sql, flags=re.IGNORECASE)
            
            # Special case: Handle complex nested patterns like JULIANDAY(decision_date) - JULIANDAY(STRFTIME(...))
            # This pattern specifically targets the error we're seeing
            adapted_sql = re.sub(
                r'JULIANDAY\s*\(([^)]+)\)\s*-\s*JULIANDAY\s*\(STRPTIME\s*\(([^)]+)\)\s*\)', 
                r'DATE_DIFF(\'day\', STRPTIME(\2), \1)', 
                adapted_sql, 
                flags=re.IGNORECASE
            )
            
            # Fix DuckDB strptime format constraint: ensure format string is a literal constant
            # Replace any strptime calls with hardcoded format for date_of_registration column
            adapted_sql = re.sub(
                r'STRPTIME\s*\(([^,]+),\s*([^)]+)\)',
                r'STRPTIME(\1, \'%d-%m-%Y\')',
                adapted_sql,
                flags=re.IGNORECASE
            )
            
        elif database_type.lower() == 'postgresql':
            # Replace DuckDB functions with PostgreSQL equivalents
            adapted_sql = adapted_sql.replace('DATE_DIFF(\'day\', ', 'EXTRACT(DAY FROM ')
            adapted_sql = adapted_sql.replace('STRPTIME(', 'TO_TIMESTAMP(')
            
        elif database_type.lower() == 'mysql':
            # Replace with MySQL equivalents
            adapted_sql = adapted_sql.replace('DATE_DIFF(\'day\', ', 'DATEDIFF(')
            adapted_sql = adapted_sql.replace('STRPTIME(', 'STR_TO_DATE(')
            
        return adapted_sql

    async def _run(self, **kwargs):
        # Extract parameters from kwargs (context-aware)
        query = kwargs.get('query')
        database_type = kwargs.get('database_type', 'duckdb')
        connection_string = kwargs.get('connection_string')
        table_info = kwargs.get('table_info')
        
        # If no connection_string provided, try to extract from database_info directly in kwargs
        if not connection_string and 'database_info' in kwargs:
            db_info = kwargs['database_info']
            if isinstance(db_info, dict):
                connection_string = db_info.get('connection')
                database_type = db_info.get('type', database_type)
        
        # If still no connection_string, try to extract from nested context
        if not connection_string and 'context' in kwargs:
            context = kwargs['context']
            if isinstance(context, dict):
                if 'connection_string' in context:
                    connection_string = context['connection_string']
                elif 's3_path' in context:
                    connection_string = context['s3_path']
                elif 'database_info' in context:
                    db_info = context['database_info']
                    if isinstance(db_info, dict):
                        connection_string = db_info.get('connection')
                        database_type = db_info.get('type', database_type)
        
        try:
            # Import required modules
            import duckdb
            from sqlalchemy import create_engine, text
            from sqlalchemy.engine import Engine
            from langchain_community.utilities.sql_database import SQLDatabase
            from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            from langchain_community.agent_toolkits.sql.base import create_sql_agent
            
            # Initialize the LLM for SQL generation
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0
            )
            
            if database_type.lower() == 'duckdb':
                # Handle DuckDB with S3 data using SQLAlchemy
                if connection_string and connection_string.startswith('s3://'):
                    # For S3 data, create a DuckDB engine with proper SQLAlchemy integration
                    db = duckdb.connect(':memory:')
                    
                    # Install and load required extensions
                    db.execute("INSTALL httpfs; LOAD httpfs;")
                    db.execute("INSTALL parquet; LOAD parquet;")
                    
                    # Set up S3 credentials if available
                    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
                    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
                    if aws_access_key and aws_secret_key:
                        db.execute(f"SET s3_region='ap-south-1';")
                        db.execute(f"SET s3_access_key_id='{aws_access_key}';")
                        db.execute(f"SET s3_secret_access_key='{aws_secret_key}';")
                    
                    # Create a generic view of the S3 data
                    s3_path = connection_string
                    db.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{s3_path}');")
                    
                    # First, inspect the schema to get actual column names
                    try:
                        schema_result = db.execute("DESCRIBE data").fetchdf()
                        sample_result = db.execute("SELECT * FROM data LIMIT 3").fetchdf()
                        
                        # Create a detailed schema description
                        schema_info = f"Available columns: {', '.join(schema_result['column_name'].tolist())}\n"
                        schema_info += f"Column types: {dict(zip(schema_result['column_name'], schema_result['column_type']))}\n"
                        schema_info += f"Sample data:\n{sample_result.to_string()}"
                        
                        # Use LLM to generate SQL query with actual schema
                        sql_prompt = f"""
                        You are a SQL expert. Generate a SQL query to answer this question: {query}
                        
                        The data is available in a view called 'data' with the following ACTUAL schema:
                        {schema_info}
                        
                        ANALYSIS TYPE GUIDANCE:
                        - For REGRESSION ANALYSIS or TREND ANALYSIS: You need individual data points, NOT aggregated by year
                          - Use SELECT without GROUP BY to get individual cases
                          - Example: SELECT date_of_registration, decision_date, year FROM data WHERE court = '33_10' AND year BETWEEN 2019 AND 2022
                          - This provides multiple rows for regression analysis
                        
                        - For COUNTING/AGGREGATION: Use GROUP BY with COUNT(), SUM(), AVG() as appropriate
                        
                        - For RAW DATA: Use simple SELECT with WHERE conditions (only when specifically requested for individual records)
                        
                        IMPORTANT: 
                        - Use ONLY the exact column names from the schema above. Do not invent or guess column names.
                        - For date calculations, use DATE_DIFF('day', start_date, end_date) instead of JULIANDAY functions
                        - For date parsing, use STRPTIME(date_string, '%d-%m-%Y') if needed
                        - Generate ONLY the SQL query, no explanations. Make sure the query is valid SQL and returns the appropriate data structure for the analysis type.
                        """
                        
                        # Get SQL from LLM
                        sql_response = await self._get_llm_response(sql_prompt)
                        
                        # Extract SQL from response (remove markdown if present)
                        sql_query = sql_response.strip()
                        if sql_query.startswith('```sql'):
                            sql_query = sql_query[7:]
                        if sql_query.endswith('```'):
                            sql_query = sql_query[:-3]
                        sql_query = sql_query.strip()
                        
                        # Additional cleaning: remove any quotes or special characters that might cause parsing errors
                        sql_query = sql_query.replace('"', '').replace('"', '').replace('"', '')
                        sql_query = sql_query.replace(''', "'").replace(''', "'")
                        
                        # Remove any trailing semicolons and clean up whitespace
                        sql_query = sql_query.rstrip(';').strip()
                        
                        # More aggressive cleaning: remove any remaining problematic characters
                        sql_query = re.sub(r'[^\w\s\(\)\[\],\.\-\+\*/<>=!\'`]', '', sql_query)
                        
                        # Validate that we have a proper SQL query
                        if not sql_query or len(sql_query) < 10:
                            return f"Error: Invalid SQL generated by LLM: {sql_response}"
                        
                        # Debug: Print the raw LLM response and cleaned SQL
                        print(f"🔍 Raw LLM response: {sql_response}")
                        print(f"🔍 Cleaned SQL: {sql_query}")
                        
                        # Adapt SQL for the specific database type
                        adapted_sql = self._adapt_sql_for_database(sql_query, database_type)
                        
                        # Debug: Print original and adapted SQL
                        print(f"Original SQL: {sql_query}")
                        print(f"Adapted SQL: {adapted_sql}")
                        
                        # Execute the SQL query directly with DuckDB
                        result = db.execute(adapted_sql).fetchdf()
                        
                        # Store the result in data cache for potential plotting tasks
                        if 'data_cache' in kwargs:
                            await kwargs['data_cache'].store("latest_sql_result", result)
                        
                        return f"Query executed successfully. Result: {result.to_dict('records')}"
                    except Exception as sql_error:
                        return f"SQL execution error: {str(sql_error)}"
                    
                else:
                    # For local DuckDB files
                    db = duckdb.connect(connection_string or ':memory:')
                    
                    # Use LLM to generate SQL query
                    sql_prompt = f"""
                    You are a SQL expert. Generate a SQL query to answer this question: {query}
                    
                    ANALYSIS TYPE GUIDANCE:
                    - For REGRESSION ANALYSIS or TREND ANALYSIS: You need individual data points, NOT aggregated by year
                      - Use SELECT without GROUP BY to get individual cases
                      - Example: SELECT date_of_registration, decision_date, year FROM data WHERE court = '33_10' AND year BETWEEN 2019 AND 2022
                      - This provides multiple rows for regression analysis
                    
                    - For COUNTING/AGGREGATION: Use GROUP BY with COUNT(), SUM(), AVG() as appropriate
                    
                    - For RAW DATA: Use simple SELECT with WHERE conditions (only when specifically requested for individual records)
                    
                    The data is available in the database. Generate ONLY the SQL query, no explanations. 
                    Make sure the query is valid DuckDB SQL and returns the appropriate data structure for the analysis type.
                    """
                    
                    # Get SQL from LLM
                    sql_response = await self._get_llm_response(sql_prompt)
                    
                    # Extract SQL from response (remove markdown if present)
                    sql_query = sql_response.strip()
                    if sql_query.startswith('```sql'):
                        sql_query = sql_query[7:]
                    if sql_query.endswith('```'):
                        sql_query = sql_query[:-3]
                    sql_query = sql_query.strip()
                    
                    # Additional cleaning: remove any quotes or special characters that might cause parsing errors
                    sql_query = sql_query.replace('"', '').replace('"', '').replace('"', '')
                    sql_query = sql_query.replace(''', "'").replace(''', "'")
                    
                    # Remove any trailing semicolons and clean up whitespace
                    sql_query = sql_query.rstrip(';').strip()
                    
                    # More aggressive cleaning: remove any remaining problematic characters
                    sql_query = re.sub(r'[^\w\s\(\)\[\],\.\-\+\*/<>=!\'`]', '', sql_query)
                    
                    # Validate that we have a proper SQL query
                    if not sql_query or len(sql_query) < 10:
                        return f"Error: Invalid SQL generated by LLM: {sql_response}"
                    
                    # Debug: Print the raw LLM response and cleaned SQL
                    print(f"🔍 Raw LLM response: {sql_response}")
                    print(f"🔍 Cleaned SQL: {sql_query}")
                    
                    # Adapt SQL for the specific database type
                    adapted_sql = self._adapt_sql_for_database(sql_query, database_type)
                    
                    # Debug: Print original and adapted SQL
                    print(f"Original SQL: {sql_query}")
                    print(f"Adapted SQL: {adapted_sql}")
                    
                    # Execute the SQL query directly with DuckDB
                    try:
                        result = db.execute(adapted_sql).fetchdf()
                        
                        # Store the result in data cache for potential plotting tasks
                        if 'data_cache' in kwargs:
                            await kwargs['data_cache'].store("latest_sql_result", result)
                        
                        return f"Query executed successfully. Result: {result.to_dict('records')}"
                    except Exception as sql_error:
                        return f"SQL execution error: {str(sql_error)}"
                    
            else:
                # For other SQL databases, use the connection string directly
                if not connection_string:
                    return "Error: Connection string required for non-DuckDB databases"
                
                # Use LLM to generate SQL query
                sql_prompt = f"""
                You are a SQL expert. Generate a SQL query to answer this question: {query}
                
                ANALYSIS TYPE GUIDANCE:
                - For REGRESSION ANALYSIS or TREND ANALYSIS: You need individual data points, NOT aggregated by year
                  - Use SELECT without GROUP BY to get individual cases
                  - Example: SELECT date_of_registration, decision_date, year FROM data WHERE court = '33_10' AND year BETWEEN 2019 AND 2022
                  - This provides multiple rows for regression analysis
                
                - For COUNTING/AGGREGATION: Use GROUP BY with COUNT(), SUM(), AVG() as appropriate
                
                - For RAW DATA: Use simple SELECT with WHERE conditions (only when specifically requested for individual records)
                
                The data is available in the database. Generate ONLY the SQL query, no explanations. 
                Make sure the query is valid SQL for the database type: {database_type} and returns the appropriate data structure for the analysis type.
                """
                
                # Get SQL from LLM
                sql_response = await self._get_llm_response(sql_prompt)
                
                # Extract SQL from response (remove markdown if present)
                sql_query = sql_response.strip()
                if sql_query.startswith('```sql'):
                    sql_query = sql_query[7:]
                if sql_query.endswith('```'):
                    sql_query = sql_query[:-3]
                sql_query = sql_query.strip()
                
                # Additional cleaning: remove any quotes or special characters that might cause parsing errors
                sql_query = sql_query.replace('"', '').replace('"', '').replace('"', '')
                sql_query = sql_query.replace(''', "'").replace(''', "'")
                
                # Remove any trailing semicolons and clean up whitespace
                sql_query = sql_query.rstrip(';').strip()
                
                # More aggressive cleaning: remove any remaining problematic characters
                sql_query = re.sub(r'[^\w\s\(\)\[\],\.\-\+\*/<>=!\'`]', '', sql_query)
                
                # Validate that we have a proper SQL query
                if not sql_query or len(sql_query) < 10:
                    return f"Error: Invalid SQL generated by LLM: {sql_response}"
                
                # Debug: Print the raw LLM response and cleaned SQL
                print(f"🔍 Raw LLM response: {sql_response}")
                print(f"🔍 Cleaned SQL: {sql_query}")
                
                # Adapt SQL for the specific database type
                adapted_sql = self._adapt_sql_for_database(sql_query, database_type)
                
                # Execute the SQL query using SQLAlchemy
                try:
                    engine = create_engine(connection_string)
                    with engine.connect() as connection:
                        result = connection.execute(text(adapted_sql))
                        rows = result.fetchall()
                        columns = result.keys()
                        
                        # Convert to list of dicts for easier handling
                        result_data = [dict(zip(columns, row)) for row in rows]
                        
                        # Store the result in data cache for potential plotting tasks
                        if 'data_cache' in kwargs:
                            await kwargs['data_cache'].store("latest_sql_result", result_data)
                        
                        return f"Query executed successfully. Result: {result_data}"
                except Exception as sql_error:
                    return f"SQL execution error: {str(sql_error)}"
                
        except Exception as e:
            return f"Error in SQL analysis: {str(e)}"

class NoSQLAnalysisInput(BaseModel):
    query: str = Field(description="The analytical question to answer about the data")
    database_type: str = Field(description="Type of NoSQL database: 'mongodb', 'redis', 'cassandra', etc.")
    connection_string: str = Field(description="Database connection string")
    collection_name: Optional[str] = Field(description="Collection/table name for the data")

class NoSQLAnalysisTool(BaseTool):
    name: str = "NoSQL Analysis Tool"
    description: str = (
        "Analyzes data from NoSQL databases like MongoDB, Redis, Cassandra, etc. "
        "This tool can handle document queries, key-value lookups, and complex aggregations. "
        "Use this when the data is stored in a NoSQL format or when you need document-based analysis."
    )
    args_schema: type[BaseModel] = NoSQLAnalysisInput

    async def _run(self, **kwargs):
        # Extract parameters from kwargs (context-aware)
        query = kwargs.get('query')
        database_type = kwargs.get('database_type', 'mongodb')
        connection_string = kwargs.get('connection_string')
        collection_name = kwargs.get('collection_name')
        
        # If no connection_string provided, try to extract from context
        if not connection_string and 'context' in kwargs:
            context = kwargs['context']
            if isinstance(context, dict):
                if 'connection_string' in context:
                    connection_string = context['connection_string']
                elif 'database_info' in context:
                    db_info = context['database_info']
                    if isinstance(db_info, dict):
                        connection_string = db_info.get('connection')
                        database_type = db_info.get('type', database_type)
        
        try:
            if database_type.lower() == 'mongodb':
                # Handle MongoDB
                from pymongo import MongoClient
                
                # Connect to MongoDB
                client = MongoClient(connection_string)
                db = client.get_default_database()
                
                if collection_name:
                    collection = db[collection_name]
                else:
                    # Get first collection if none specified
                    collection = db.list_collection_names()[0]
                    collection = db[collection]
                
                # Create a simple agent with Python REPL for MongoDB operations
                tools = [PythonREPLTool()]
                agent = create_react_agent(llm=ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0
                ), tools=tools, verbose=True)
                
                # Create a context-aware prompt with MongoDB context
                context_prompt = f"""
                You have access to a MongoDB collection. Use Python with pymongo to answer this question: {query}
                
                Available collection: {collection.name}
                Database: {db.name}
                
                The collection is already connected and available. Use the Python REPL tool to write and execute code that answers the question.
                You can use pymongo operations like find(), aggregate(), count_documents(), etc.
                """
                
                result = agent.invoke({"input": context_prompt})
                return result["output"]
                
            elif database_type.lower() == 'redis':
                # Handle Redis
                import redis
                
                # Connect to Redis
                r = redis.from_url(connection_string)
                
                # Create agent with Python REPL
                tools = [PythonREPLTool()]
                agent = create_react_agent(llm=ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0
                ), tools=tools, verbose=True)
                
                context_prompt = f"""
                You have access to a Redis database. Use Python with redis to answer this question: {query}
                
                Use the Python REPL tool to write and execute code that answers the question.
                """
                
                result = agent.invoke({"input": context_prompt})
                return result["output"]
                
            else:
                return f"Unsupported NoSQL database type: {database_type}. Supported types: mongodb, redis"
                
        except Exception as e:
            return f"Error in NoSQL analysis: {str(e)}"

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

async def create_planner_executor_workflow(user_query: str, file_path: Optional[str] = None):
    """
    Creates and executes a workflow using the new Planner-Executor architecture.
    This handles complex, multi-part queries with intelligent task planning and parallel execution.
    """
    try:
        # Initialize the LLM for planning
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )
        
        # Initialize the data cache for sharing data between tasks
        data_cache = DataCache()
        
        # Create the Planner Agent to generate execution plan
        planner = PlannerAgent(llm)
        
        print("🧠 Creating execution plan...")
        execution_plan = await planner.create_execution_plan(user_query, file_path)
        
        print(f"📋 Execution plan created with {len(execution_plan.stages)} stages")
        for stage in execution_plan.stages:
            print(f"  Stage {stage.stage_id}: {stage.description} ({len(stage.tasks)} tasks)")
            for task in stage.tasks:
                print(f"    - {task.task_id}: {task.description}")
        
        # Create the Executor Crew to execute the plan
        # Initialize with all available tools
        tools = {
            "ScrapingTool": ScrapingTool(),
            "DataframeAnalysisTool": DataframeAnalysisTool(),
            "PDFAnalysisTool": PDFAnalysisTool(),
            "SQLAnalysisTool": SQLAnalysisTool(llm=llm),
            "NoSQLAnalysisTool": NoSQLAnalysisTool()
        }
        executor = ExecutorCrew(data_cache, tools)
        
        print("🚀 Executing plan...")
        results = await executor.execute_plan(execution_plan)
        
        # Format the results for the user
        final_results = {}
        for task_id, result in results.items():
            if isinstance(result, dict) and "error" in result:
                final_results[task_id] = f"Error: {result['error']}"
            else:
                final_results[task_id] = result
        
        # Create a summary response
        if len(final_results) == 1:
            # Single result, return it directly
            result = list(final_results.values())[0]
            if isinstance(result, str):
                return result, ["Planner-Executor workflow completed successfully"]
            else:
                return str(result), ["Planner-Executor workflow completed successfully"]
        else:
            # Multiple results, format them nicely
            summary = "Analysis completed successfully. Here are the results:\n\n"
            for task_id, result in final_results.items():
                summary += f"**{task_id}**:\n{result}\n\n"
            return summary, ["Planner-Executor workflow completed successfully"]
            
    except Exception as e:
        print(f"❌ Planner-Executor workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Planner-Executor workflow failed: {str(e)}"}, []

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
            "- Use the Dataframe Analysis Tool when you have tabular data (like from a CSV) and the query requires specific numerical calculations, statistical analysis, or visualizations that you cannot perform natively.\n"
            "- Use the SQL Analysis Tool for complex analytical questions that require SQL operations on databases (including DuckDB for S3 data, PostgreSQL, MySQL, etc.). This is ideal for aggregations, joins, and statistical analysis.\n"
            "- Use the NoSQL Analysis Tool for document-based databases like MongoDB, Redis, or when you need to analyze unstructured or semi-structured data."
        ),
        tools=[ScrapingTool(), DataframeAnalysisTool(), PDFAnalysisTool(), SQLAnalysisTool(), NoSQLAnalysisTool()],
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

        # Execute the workflow using the new Planner-Executor architecture
        result, reasoning_steps = await create_planner_executor_workflow(final_query, file_path=temp_file_path)
        
        # Parse the final result correctly
        if isinstance(result, dict):
            if "error" in result:
                final_answer = f"Error: {result['error']}"
            elif "answer" in result:
                final_answer = result["answer"]
            elif "result" in result:
                final_answer = result["result"]
            else:
                final_answer = str(result)
        elif hasattr(result, 'raw'):
            final_answer = result.raw
        elif hasattr(result, 'output'):
            final_answer = result.output
        elif isinstance(result, str):
            final_answer = result
        else:
            final_answer = str(result)
        
        return AgentResponse(
            success=True,
            result={"answer": final_answer},
            message="Successfully executed Planner-Executor workflow.",
            agent_used="planner_executor",
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

@app.post("/analyze_file", response_model=AgentResponse)
async def analyze_file(file: UploadFile = File(...)):
    """
    This endpoint takes a text file containing multiple questions and runs the 
    Planner-Executor workflow to handle complex, multi-part queries.
    """
    temp_file_path = None
    try:
        # Clean up any existing temp files before starting
        cleanup_temp_files()
        
        if not file.filename.endswith('.txt'):
            return AgentResponse(
                success=False,
                result={"error": "Only .txt files are supported"},
                message="Please upload a .txt file containing your questions",
                agent_used="none"
            )
        
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read the content of the text file
        with open(temp_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        
        # Execute the Planner-Executor workflow with the file content
        result, reasoning_steps = await create_planner_executor_workflow(file_content, file_path=temp_file_path)
        
        # Parse the final result
        if isinstance(result, dict):
            if "error" in result:
                final_answer = f"Error: {result['error']}"
            elif "answer" in result:
                final_answer = result["answer"]
            elif "result" in result:
                final_answer = result["result"]
            else:
                final_answer = str(result)
        else:
            final_answer = str(result)
        
        return AgentResponse(
            success=True,
            result={"answer": final_answer},
            message="Successfully executed Planner-Executor workflow on text file.",
            agent_used="planner_executor",
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

