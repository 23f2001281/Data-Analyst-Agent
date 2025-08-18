from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from crewai.tools import BaseTool
import re
import asyncio
from asyncio import TimeoutError
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
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
import sqlalchemy
from sqlalchemy import create_engine
from dataclasses import dataclass
from datetime import datetime
from response_formatter import ResponseFormatter
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set to non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import base64

load_dotenv()

app = FastAPI(title="Data Analyst AI Agent")

# Debug logging control (set AGENT_DEBUG=true to enable verbose logs)
DEBUG = os.getenv("AGENT_DEBUG", "false").lower() == "true"

def dprint(*args, **kwargs) -> None:
    if DEBUG:
        print(*args, **kwargs)

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
        # CRITICAL FIX: Store file_path for later use in context
        self._file_path = file_path
        if file_path:
            dprint(f"🔍 PlannerAgent: Stored file_path: {file_path}")
        
        # Create the planning prompt
        planning_prompt = await self._create_planning_prompt(user_query, file_path)
        
        max_retries = 1
        for attempt in range(max_retries):
            try:
                dprint(f"🤖 Planning prompt created, length: {len(planning_prompt)}")
                dprint(f"🤖 Attempt {attempt + 1}/{max_retries}")
                
                # Get the LLM response for planning
                response = await self._get_llm_response(planning_prompt)
                dprint(f"🤖 LLM response received, length: {len(response)}")
                dprint(f"🤖 LLM response preview: {response[:200]}...")
                
                # Parse the response into an execution plan
                execution_plan = self._parse_planning_response(response)
                dprint(f"✅ Execution plan created successfully with {len(execution_plan.stages)} stages")
                
                return execution_plan
                
            except Exception as e:
                dprint(f"❌ Planning attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    dprint(f"🔄 Retrying with simplified prompt...")
                    # Try with an even simpler prompt on retry
                    simple_prompt = self._create_simple_planning_prompt(user_query, file_path)
                    planning_prompt = simple_prompt
                else:
                    dprint(f"❌ All planning attempts failed, raising error")
                    raise
        
        # This should never be reached, but just in case
        raise Exception("Failed to create execution plan after all retry attempts")
    
    async def _create_planning_prompt(self, user_query: str, file_path: Optional[str] = None) -> str:
        """Creates a focused prompt for the LLM to generate an execution plan using ChatPromptTemplate"""
        
        # Use LLM to intelligently parse and clean the user query
        clean_query = await self._parse_user_query_with_llm(user_query)
        
        # Create structured prompt template
        system_template = """You are an AI task planner. Create execution plans for data analysis requests.

PLANNING RULES:
1. For CSV files: Use DataframeAnalysisTool directly (no data acquisition needed)
2. For database analysis: Use SQLAnalysisTool first, then DataframeAnalysisTool for visualization
3. For web scraping: Use ScrapingTool first, then DataframeAnalysisTool for analysis
4. For PDFs: Use PDFAnalysisTool directly
5. **CRITICAL: NEVER create multiple DataframeAnalysisTool tasks - use only ONE for final analysis**
6. If the data source is a database, you MUST include database_info with both type and connection
   - Extract connection details (e.g., S3 paths, JDBC URLs) from the RAW INPUT if present
   - If an S3 path for parquet files is present, set type to "duckdb" and use that S3 path as connection

Return ONLY valid JSON. No explanations outside the JSON object."""

        human_template = """Create an execution plan for this request:

USER REQUEST (cleaned):
{clean_query}

RAW INPUT (for extracting connection details if present; do not include raw text in output):
{raw_input}

{file_info}Available tools: ScrapingTool, DataframeAnalysisTool, PDFAnalysisTool, SQLAnalysisTool, NoSQLAnalysisTool

RETURN FORMAT:
{{
  "context": {{
    "data_source_type": "file|database|web_scraping|pdf",
    "user_query": "{clean_query}",
    "file_path": "{file_path}",
    "database_info": {{"type": "database_type", "connection": "connection_string"}}
  }},
  "stages": [
    {{
      "stage_id": 1,
      "description": "Data acquisition",
      "tasks": [
        {{
          "task_id": "unique_id",
          "tool": "tool_name",
          "description": "What to do",
          "dependencies": [],
          "output_id": "output_name"
        }}
      ]
    }},
    {{
      "stage_id": 2,
      "description": "Analysis",
      "tasks": [
        {{
          "task_id": "unique_id",
          "tool": "tool_name", 
          "description": "What to do",
          "dependencies": ["stage1_task_id"],
          "output_id": "output_name"
        }}
      ]
    }}
  ]
}}"""

        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        # Format the prompt with variables
        file_info = f"FILE: {file_path}\n" if file_path else ""
        raw_input = user_query[:2000]
        
        formatted_prompt = prompt_template.format_messages(
            clean_query=clean_query,
            raw_input=raw_input,
            file_info=file_info,
            file_path=file_path if file_path else ""
        )
        
        # Convert to string format for the LLM
        prompt_text = ""
        for message in formatted_prompt:
            if message.type == "system":
                prompt_text += f"SYSTEM: {message.content}\n\n"
            elif message.type == "human":
                prompt_text += f"USER: {message.content}\n\n"
        
        return prompt_text
    
    async def _parse_user_query_with_llm(self, user_query: str) -> str:
        """Use LLM to intelligently parse and format the user query for JSON safety"""
        try:
            # Create structured prompt template for query parsing
            system_template = """Parse user queries and return clean, structured versions that can be safely embedded in JSON.

Requirements:
- Remove any markdown formatting, code blocks, or special characters
- Extract the core questions/requests clearly
- Make it JSON-safe (no unescaped quotes, newlines, etc.)
- Keep it concise but complete
- Focus on the essential information needed for task planning
- Return ONLY the cleaned query, no explanations or markdown"""

            human_template = """Parse this user query and return a clean, structured version that can be safely embedded in JSON.

Original query:
{user_query}

Cleaned query:"""

            # Create the prompt template
            parsing_prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
            
            # Format the prompt
            formatted_parsing_prompt = parsing_prompt_template.format_messages(user_query=user_query)
            
            # Convert to string format for the LLM
            parsing_prompt = ""
            for message in formatted_parsing_prompt:
                if message.type == "system":
                    parsing_prompt += f"SYSTEM: {message.content}\n\n"
                elif message.type == "human":
                    parsing_prompt += f"USER: {message.content}\n\n"
            
            llm = llm_manager.get_llm(temperature=0.0)
            response = await llm.ainvoke(parsing_prompt)
            cleaned = response.content.strip()
            
            dprint(f"🔍 LLM parsed user query from {len(user_query)} to {len(cleaned)} chars")
            dprint(f"🔍 Cleaned query: {cleaned[:200]}...")
            
            return cleaned
            
        except Exception as e:
            dprint(f"⚠️ LLM query parsing failed: {e}, using fallback cleaning")
            return self._clean_user_query_for_json_fallback(user_query)
    
    def _clean_user_query_for_json_fallback(self, query: str) -> str:
        """Fallback manual cleaning if LLM parsing fails"""
        if not query:
            return ""
        
        cleaned = (query
            .replace('"', '\\"')  # Escape quotes
            .replace('\n', ' ')   # Replace newlines with spaces
            .replace('\t', ' ')   # Replace tabs with spaces
            .replace('\r', ' ')   # Replace carriage returns
            .strip())             # Remove leading/trailing whitespace
        
        # Remove multiple spaces
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        dprint(f"🔍 Fallback cleaned user query from {len(query)} to {len(cleaned)} chars")
        return cleaned
    
    async def _create_simple_planning_prompt(self, user_query: str, file_path: Optional[str] = None) -> str:
        """Creates a very simple fallback prompt for the LLM using ChatPromptTemplate"""
        
        # Use LLM to intelligently parse and clean the user query
        clean_query = await self._parse_user_query_with_llm(user_query)
        
        system_template = """Create a simple execution plan for data analysis requests.

PLANNING RULES:
1. For CSV files: Use DataframeAnalysisTool directly
2. For database analysis: Use SQLAnalysisTool first, then DataframeAnalysisTool
3. For web scraping: Use ScrapingTool first, then DataframeAnalysisTool
4. For PDFs: Use PDFAnalysisTool directly
5. **CRITICAL: NEVER create multiple DataframeAnalysisTool tasks - use only ONE for final analysis**
6. If an S3 path for parquet files is present, set type to "duckdb" and use that S3 path as connection"""

        human_template = """Create a simple execution plan for: {clean_query}

RAW INPUT (for extracting connection details if present; do not include raw text in output):
{raw_input}

{file_info}Available tools: ScrapingTool, DataframeAnalysisTool, PDFAnalysisTool, SQLAnalysisTool, NoSQLAnalysisTool

Return this JSON structure:
{{
  "context": {{
    "data_source_type": "database",
    "user_query": "{clean_query}",
    "file_path": "{file_path}",
    "database_info": {{"type": "database_type", "connection": "connection_string"}}
  }},
  "stages": [
    {{
      "stage_id": 1,
      "description": "Get data",
      "tasks": [
        {{
          "task_id": "get_data",
          "tool": "SQLAnalysisTool",
          "description": "Get data from DuckDB S3 parquet files",
          "dependencies": [],
          "output_id": "data"
        }}
      ]
    }},
    {{
      "stage_id": 2,
      "description": "Analyze data",
      "tasks": [
        {{
          "task_id": "analyze_data",
          "tool": "DataframeAnalysisTool",
          "description": "Analyze the data",
          "dependencies": ["get_data"],
          "output_id": "analysis"
        }}
      ]
    }}
  ]
}}"""

        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        # Format the prompt with variables
        file_info = f"File: {file_path}\n" if file_path else ""
        raw_input = user_query[:1200]
        
        formatted_prompt = prompt_template.format_messages(
            clean_query=clean_query,
            raw_input=raw_input,
            file_info=file_info,
            file_path=file_path if file_path else ""
        )
        
        # Convert to string format for the LLM
        prompt_text = ""
        for message in formatted_prompt:
            if message.type == "system":
                prompt_text += f"SYSTEM: {message.content}\n\n"
            elif message.type == "human":
                prompt_text += f"USER: {message.content}\n\n"
        
        return prompt_text
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Gets response from the LLM for planning"""
        try:
            # Use centralized LLM manager
            llm = llm_manager.get_llm(temperature=0.0)
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            dprint(f"❌ LLM planning request failed: {str(e)}")
            raise
    
    def _parse_planning_response(self, response: str) -> ExecutionPlan:
        """Parses the LLM response into an ExecutionPlan object"""
        try:
            # Clean the response to extract just the JSON
            response = response.strip()
            
            # Remove markdown code blocks
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Try to find JSON content if it's embedded in other text
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            dprint(f"🔍 Cleaned response for parsing: {response[:200]}...")
            
            # Parse the JSON
            plan_data = json.loads(response)
            
            dprint(f"🔍 Parsed plan_data keys: {list(plan_data.keys())}")
            if 'context' in plan_data:
                dprint(f"🔍 Context keys: {list(plan_data['context'].keys())}")
                if 'database_info' in plan_data['context']:
                    dprint(f"🔍 database_info: {plan_data['context']['database_info']}")
            
            # Validate required fields
            if 'stages' not in plan_data:
                raise ValueError("Missing 'stages' field in planning response")
            
            if 'context' not in plan_data:
                raise ValueError("Missing 'context' field in planning response")
            
            # Convert to ExecutionPlan object
            stages = []
            for stage_data in plan_data.get("stages", []):
                if 'stage_id' not in stage_data or 'tasks' not in stage_data:
                    dprint(f"⚠️ Skipping invalid stage: {stage_data}")
                    continue
                    
                tasks = []
                for task_data in stage_data.get("tasks", []):
                    if 'task_id' not in task_data or 'tool' not in task_data:
                        dprint(f"⚠️ Skipping invalid task: {task_data}")
                        continue
                        
                    task = ExecutionTask(
                        task_id=task_data["task_id"],
                        tool=task_data.get("tool", ""),
                        description=task_data.get("description", ""),
                        dependencies=task_data.get("dependencies", []),
                        output_id=task_data.get("output_id"),
                        context=task_data.get("context", {})
                    )
                    tasks.append(task)
                
                if tasks:  # Only add stage if it has valid tasks
                    stage = ExecutionStage(
                        stage_id=stage_data["stage_id"],
                        description=stage_data.get("description", ""),
                        tasks=tasks,
                        can_run_parallel=stage_data.get("can_run_parallel", True)
                    )
                    stages.append(stage)
            
            if not stages:
                raise ValueError("No valid stages found in planning response")
            
            # CRITICAL FIX: Include file_path in the execution plan context
            context = plan_data.get("context", {})
            if hasattr(self, '_file_path') and self._file_path:
                context['file_path'] = self._file_path
                dprint(f"🔍 PlannerAgent: Added file_path to execution plan context: {self._file_path}")
            
            # Fallback: If database_info missing but S3 path present in user_query, infer DuckDB connection
            try:
                if 'database_info' not in context:
                    user_q = context.get('user_query', '') or ''
                    import re
                    s3_match = re.search(r"s3://[^\s'\"]+", user_q)
                    if s3_match:
                        inferred_conn = s3_match.group(0)
                        context['database_info'] = {"type": "duckdb", "connection": inferred_conn}
                        dprint(f"🔍 PlannerAgent: Inferred database_info from user_query: {context['database_info']}")
            except Exception as infer_err:
                dprint(f"⚠️ PlannerAgent: Failed to infer database_info: {infer_err}")
            
            return ExecutionPlan(
                context=context,
                stages=stages,
                metadata=plan_data.get("metadata", {})
            )
            
        except Exception as e:
            dprint(f"❌ Failed to parse planning response: {str(e)}")
            dprint(f"Raw response: {response}")
            raise

class ExecutorCrew:
    """Executes the execution plan created by the Planner Agent"""
    
    def __init__(self, data_cache: DataCache, tools: Dict[str, Any]):
        self.data_cache = data_cache
        self.tools = tools
        self.execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def execute_plan(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Executes the execution plan stage by stage, handling dependencies and parallel execution
        """
        dprint(f"🚀 Executing execution plan with {len(execution_plan.stages)} stages")
        dprint(f"🚀 Execution plan context keys: {list(execution_plan.context.keys())}")
        dprint(f"🚀 Execution plan context: {execution_plan.context}")
        
        # DEBUG: Show the actual execution plan structure
        for i, stage in enumerate(execution_plan.stages):
            dprint(f"🚀 Stage {i+1}: {stage.description}")
            for j, task in enumerate(stage.tasks):
                dprint(f"🚀   Task {j+1}: {task.task_id} -> tool: {task.tool} -> deps: {task.dependencies}")
                dprint(f"🚀     Description: {task.description}")
                dprint(f"🚀     Context: {task.context}")
        
        # Store the execution plan context for tools to access
        execution_plan_context = execution_plan.context.copy()
        self.execution_plan_context = execution_plan_context
        
        dprint(f"🔍 ExecutorCrew: Stored execution plan context")
        dprint(f"🔍 ExecutorCrew: Full execution plan context keys: {list(execution_plan_context.keys())}")
        
        results = {}
        failed_tasks = []
        
        # CRITICAL FIX: Execute stages sequentially and ensure each stage completes before moving to the next
        # Also track which tools have been executed to prevent duplicates
        executed_tools = set()
        
        for stage in execution_plan.stages:
            dprint(f"📋 EXECUTING Stage {stage.stage_id}: {stage.description}")
            dprint(f"📋 Stage {stage.stage_id} has {len(stage.tasks)} tasks")
            
            # Execute all tasks in this stage
            stage_results = {}
            for task in stage.tasks:
                dprint(f"📝 EXECUTING task: {task.task_id} with tool: {task.tool}")
                
                # CRITICAL: Prevent duplicate tool execution
                if task.tool in executed_tools:
                    dprint(f"⚠️ Tool {task.tool} already executed, skipping {task.task_id}")
                    stage_results[task.task_id] = f"Tool {task.tool} already executed - duplicate task skipped"
                    continue
                
                try:
                    # Execute the task
                    result = await self._execute_single_task(task, results, stage_results)
                    stage_results[task.task_id] = result
                    
                    # Mark this tool as executed
                    executed_tools.add(task.tool)
                    
                    # Store output in cache if specified
                    if task.output_id:
                        await self.data_cache.store(task.output_id, result)
                    
                    dprint(f"✅ Task {task.task_id} completed successfully")
                    
                except Exception as e:
                    dprint(f"❌ Task {task.task_id} failed: {str(e)}")
                    stage_results[task.task_id] = {"error": str(e)}
                    failed_tasks.append(task.task_id)
                    fallback_manager.record_task_failure(task.task_id, str(e), {
                        'stage_id': stage.stage_id,
                        'execution_id': self.execution_id
                    })
            
            # CRITICAL: Store stage results BEFORE moving to next stage
            results.update(stage_results)
            dprint(f"✅ Stage {stage.stage_id} completed with {len(stage_results)} tasks")
            dprint(f"📊 Current results keys: {list(results.keys())}")
            
            # CRITICAL: Verify Stage 1 actually completed and has data
            if stage.stage_id == 1:
                dprint(f"🔍 Stage 1 completed, verifying data availability...")
                stage1_tasks = [task.task_id for task in stage.tasks]
                dprint(f"🔍 Stage 1 tasks: {stage1_tasks}")
                
                for task_id in stage1_tasks:
                    if task_id in results:
                        result = results[task_id]
                        dprint(f"🔍 Stage 1 task {task_id} result: {type(result)} - {str(result)[:200]}...")
                        
                        # Check if this is an error result
                        if isinstance(result, str) and 'Error:' in result:
                            dprint(f"❌ Stage 1 task {task_id} FAILED with error: {result}")
                            # Don't continue to Stage 2 if Stage 1 failed
                            dprint(f"❌ Stopping execution - Stage 1 failed")
                            return results
                    else:
                        dprint(f"❌ Stage 1 task {task_id} not found in results")
                
                dprint(f"✅ Stage 1 completed successfully, proceeding to Stage 2")
            
            # CRITICAL: Don't proceed to next stage if current stage had failures
            if failed_tasks:
                dprint(f"❌ Stage {stage.stage_id} had failures, stopping execution")
                break
        
        # Check if we need to re-plan due to failures
        if failed_tasks and fallback_manager.should_replan(self.execution_id):
            dprint(f"🔄 Execution had {len(failed_tasks)} failed tasks, attempting re-planning...")
            return await self._attempt_replanning(execution_plan, failed_tasks, results)
        
        # If we have failures but can't re-plan, record execution failure
        if failed_tasks:
            fallback_manager.record_execution_failure(self.execution_id, 
                f"Execution completed with {len(failed_tasks)} failed tasks", failed_tasks)
        
        return results
    
    
    async def _execute_single_task(self, task: ExecutionTask, previous_results: Dict[str, Any], current_stage_results: Dict[str, Any]) -> Any:
        """Executes a single task using the appropriate tool"""
        
        # Get the tool
        tool_name = task.tool
        
        # Validate tool selection
        if not tool_name:
            # Use intelligent tool guessing only when no tool is specified
            tool_name = self._guess_tool_from_task(task)
            dprint(f"🔍 ExecutorCrew: No tool specified, guessed: {tool_name}")
        else:
            dprint(f"🔍 ExecutorCrew: Using specified tool: {tool_name}")
        
        dprint(f"🔍 ExecutorCrew: Final tool selected: {tool_name}")
        
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        
        # Prepare context for the tool
        context = self._prepare_task_context(task, previous_results, current_stage_results)

        # Pass custom_format if it exists in the execution plan context
        if hasattr(self, 'execution_plan_context') and 'custom_format' in self.execution_plan_context:
            context['custom_format'] = self.execution_plan_context['custom_format']

        # Optional routing: only send relevant sub-questions to specific tools
        try:
            routing_enabled = os.getenv("ROUTING_ENABLED", "true").lower() == "true"
            data_source_type = context.get("data_source_type", "")

            # Bypass routing if the data source is a file and not a database
            if routing_enabled and data_source_type != "database" and tool_name == "DataframeAnalysisTool":
                dprint(f"🔍 ExecutorCrew: Bypassing routing for file-based data source with {tool_name}")
            elif routing_enabled and isinstance(context.get("query"), str):
                if tool_name == "DataframeAnalysisTool":
                    filtered_query = await self._filter_query_for_tool(context["query"], tool_name)
                    if not filtered_query.strip():
                        dprint(f"🔍 ExecutorCrew: Routing skipped {tool_name} - no visualization questions detected")
                        return "No DataFrame/plot questions detected; nothing to analyze."
                    context["query"] = filtered_query
                elif tool_name == "SQLAnalysisTool":
                    # Optionally route only SQL/both questions to SQLAnalysisTool
                    filtered_query = await self._filter_query_for_tool(context["query"], tool_name)
                    if filtered_query.strip():
                        context["query"] = filtered_query
        except Exception as route_err:
            dprint(f"⚠️ ExecutorCrew: Routing step failed, proceeding without routing: {route_err}")
        
        dprint(f"🔍 ExecutorCrew: Executing {task.task_id} with context keys: {list(context.keys())}")
        dprint(f"🔍 ExecutorCrew: Query being passed: {context.get('query', 'No query')[:200]}...")
        
        # Execute the tool
        if hasattr(tool, '_run'):
            result = await tool._run(**context)
        else:
            # Fallback for sync tools
            result = tool._run(**context)
        
        dprint(f"🔍 ExecutorCrew: Task {task.task_id} completed with result type: {type(result)}")
        if isinstance(result, str):
            dprint(f"🔍 ExecutorCrew: Task {task.task_id} result preview: {result[:200]}...")
        
        return result
    
    def _guess_tool_from_task(self, task: ExecutionTask) -> str:
        """Guess the appropriate tool based on task ID and description"""
        task_id = task.task_id.lower()
        description = task.description.lower() if task.description else ""
        
        # SQL-related tasks
        if any(keyword in task_id for keyword in ['sql', 'data', 'fetch', 'get', 's3', 'court', 'database']):
            if any(keyword in description for keyword in ['sql', 'query', 'database', 'duckdb', 's3']):
                return "SQLAnalysisTool"
        
        # Data analysis tasks
        if any(keyword in task_id for keyword in ['analyze', 'analysis', 'plot', 'chart', 'regression']):
            if any(keyword in description for keyword in ['plot', 'chart', 'visualization', 'analysis', 'pandas']):
                return "DataframeAnalysisTool"
        
        # Web scraping tasks
        if any(keyword in task_id for keyword in ['scrape', 'web', 'url', 'html']):
            if any(keyword in description for keyword in ['scrape', 'web', 'url', 'html']):
                return "ScrapingTool"
        
        # PDF/document tasks
        if any(keyword in task_id for keyword in ['pdf', 'document', 'doc']):
            if any(keyword in description for keyword in ['pdf', 'document', 'extract']):
                return "PDFAnalysisTool"
        
        # Default to DataframeAnalysisTool for analysis tasks
        if any(keyword in task_id for keyword in ['analyze', 'analysis', 'answer', 'question']):
            return "DataframeAnalysisTool"
        
        # Default to SQLAnalysisTool for data tasks
        if any(keyword in task_id for keyword in ['data', 'fetch', 'get']):
            return "SQLAnalysisTool"
        
        # Final fallback
        return "DataframeAnalysisTool"
    
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
        
        # CRITICAL FIX: Preserve the original user query instead of generic task description
        # The original user query contains the specific questions that need to be answered
        if "query" not in context:
            # Check if we have the original user query in execution plan context
            if hasattr(self, 'execution_plan_context') and 'user_query' in self.execution_plan_context:
                context["query"] = self.execution_plan_context['user_query']
                dprint(f"🔍 ExecutorCrew: Using original user query: {context['query'][:200]}...")
            else:
                # Fallback to task description if no original query available
                context["query"] = task.description
                dprint(f"⚠️ ExecutorCrew: No original query found, using task description: {context['query']}")
        else:
            dprint(f"🔍 ExecutorCrew: Query already present in context: {context['query'][:200]}...")
        
        # CRITICAL FIX: Pass file_path from execution plan context to tools
        if hasattr(self, 'execution_plan_context') and 'file_path' in self.execution_plan_context:
            context["file_path"] = self.execution_plan_context['file_path']
            dprint(f"🔍 ExecutorCrew: Added file_path to context: {context['file_path']}")
        
        # Debug: Show what query is being passed to the tool
        dprint(f"🔍 ExecutorCrew: Final query being passed to {task.tool}: {context['query'][:200]}...")
        
        return context

    async def _filter_query_for_tool(self, query_text: str, tool_name: str) -> str:
        """Split the user's query into sub-questions and select only those relevant to the tool.
        tool_name: 'DataframeAnalysisTool' or 'SQLAnalysisTool'"""
        questions = self._split_questions(query_text)
        if not questions:
            return query_text

        try:
            labels = await self._classify_questions(questions)
        except Exception as _:
            labels = self._classify_heuristic(questions)

        selected: list[str] = []
        for idx, q in enumerate(questions):
            label = labels.get(idx, "both")
            if tool_name == "DataframeAnalysisTool" and label in ("dataframe_plot", "both"):
                selected.append(q)
            if tool_name == "SQLAnalysisTool" and label in ("sql_only", "both"):
                selected.append(q)

        return "\n\n".join(selected)

    def _split_questions(self, query_text: str) -> list[str]:
        """Split multi-question text into atomic questions conservatively."""
        if not query_text:
            return []
        lines = [ln.strip() for ln in str(query_text).splitlines() if ln.strip()]
        # Group lines into numbered bullets or question-mark endings
        import re as _re
        buckets: list[str] = []
        current: list[str] = []
        bullet_pattern = _re.compile(r"^\s*(?:\d+\.|\d+\)|[-*])\s+")
        for ln in lines:
            if bullet_pattern.match(ln) or ln.lower().startswith("question"):
                if current:
                    buckets.append(" ".join(current).strip())
                    current = []
                # strip the bullet
                ln = bullet_pattern.sub("", ln)
                current.append(ln)
            else:
                current.append(ln)
        if current:
            buckets.append(" ".join(current).strip())

        # If we couldn't detect multiple, fallback to naive split by '?'
        if len(buckets) <= 1 and "?" in query_text:
            naive = [seg.strip()+"?" for seg in query_text.split("?") if seg.strip()]
            return naive or buckets
        return buckets or [query_text]

    async def _classify_questions(self, questions: list[str]) -> dict[int, str]:
        """Use LLM to classify each question into sql_only | dataframe_plot | both.
        Returns mapping index->label."""
        # Small, robust prompt to minimize cost
        sys_t = (
            "Classify each question as: sql_only, dataframe_plot, or both. "
            "Return ONLY JSON array of objects: {index: number, label: string}."
        )
        content_lines = []
        for i, q in enumerate(questions):
            content_lines.append(f"{i}. {q}")
        human_t = "\n".join(content_lines)

        prompt = f"SYSTEM: {sys_t}\n\nUSER: {human_t}\n\n"
        llm = llm_manager.get_llm(temperature=0.0)
        resp = await llm.ainvoke(prompt)
        text = resp.content.strip()
        # Clean code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]

        import json as _json
        data = _json.loads(text)
        labels: dict[int, str] = {}
        if isinstance(data, list):
            for item in data:
                try:
                    idx = int(item.get("index"))
                    lab = str(item.get("label", "both")).strip().lower()
                    # Ensure plotting questions also go to SQL to fetch row-level data
                    if lab == "dataframe_plot":
                        lab = "both"
                    if lab not in ("sql_only", "dataframe_plot", "both"):
                        lab = "both"
                    labels[idx] = lab
                except Exception:
                    continue
        return labels

    def _classify_heuristic(self, questions: list[str]) -> dict[int, str]:
        """Fallback heuristic routing without LLM."""
        viz_keywords = [
            "plot", "chart", "visualize", "graph", "scatter", "line plot", "bar", "histogram",
            "heatmap", "regression plot", "trend", "distribution", "image", "base64", "figure"
        ]
        labels: dict[int, str] = {}
        for i, q in enumerate(questions):
            ql = q.lower()
            is_viz = any(k in ql for k in viz_keywords)
            # Route viz questions to BOTH SQL (to fetch rows) and DataFrame (to plot)
            labels[i] = "both" if is_viz else "sql_only"
        return labels

    async def _attempt_replanning(self, original_plan: ExecutionPlan, failed_tasks: list, previous_results: dict) -> Dict[str, Any]:
        """Attempts to re-plan execution when tasks fail"""
        dprint(f"🔄 Attempting re-planning for failed tasks: {failed_tasks}")
        
        try:
            # Analyze failures to understand what went wrong
            failure_analysis = {}
            for task_id in failed_tasks:
                analysis = fallback_manager.analyze_failures(task_id=task_id)
                failure_analysis[task_id] = analysis
            
            # Create a new plan that avoids the failed tasks or uses alternative approaches
            new_plan = await self._create_recovery_plan(original_plan, failed_tasks, failure_analysis, previous_results)
            
            if new_plan:
                dprint(f"🔄 Re-planning successful, executing recovery plan...")
                # Execute the recovery plan
                recovery_results = await self._execute_recovery_plan(new_plan, previous_results)
                
                # Merge results
                all_results = {**previous_results, **recovery_results}
                return all_results
            else:
                dprint(f"🔄 Re-planning failed, returning partial results with failure information")
                # Add failure analysis to results
                for task_id in failed_tasks:
                    if task_id not in previous_results:
                        previous_results[task_id] = {
                            'error': 'Task failed and could not be recovered',
                            'failure_analysis': failure_analysis.get(task_id, {})
                        }
                return previous_results
                
        except Exception as e:
            dprint(f"❌ Re-planning failed with error: {str(e)}")
            fallback_manager.record_execution_failure(self.execution_id, f"Re-planning failed: {str(e)}", failed_tasks)
            return previous_results
    
    async def _create_recovery_plan(self, original_plan: ExecutionPlan, failed_tasks: list, failure_analysis: dict, previous_results: dict) -> Optional[ExecutionPlan]:
        """Creates a recovery plan to handle failed tasks"""
        try:
            # Get the original user query from the execution plan context
            user_query = self.execution_plan_context.get('user_query', '')
            
            # Create a recovery prompt for the planner
            recovery_prompt = f"""
            ORIGINAL QUERY: {user_query}
            
            EXECUTION FAILED: The following tasks failed during execution:
            {json.dumps(failed_tasks, indent=2)}
            
            FAILURE ANALYSIS:
            {json.dumps(failure_analysis, indent=2)}
            
            PREVIOUS RESULTS (successful tasks):
            {json.dumps(list(previous_results.keys()), indent=2)}
            
            TASK: Create a recovery execution plan that:
            1. Avoids the failed tasks or uses alternative approaches
            2. Builds upon successful previous results
            3. Still answers the original user query
            4. Uses different tools or strategies for failed tasks
            
            IMPORTANT: Focus on completing the user's request despite the failures.
            """
            
            # Use centralized LLM manager for recovery plan
            llm = llm_manager.get_llm(temperature=0.0)
            
            # Create a simple recovery planner
            recovery_response = await llm.ainvoke(recovery_prompt)
            
            # Parse the recovery plan (this is a simplified approach)
            # In a full implementation, you'd want to parse this into a proper ExecutionPlan
            dprint(f"🔄 Recovery plan generated: {recovery_response.content[:500]}...")
            
            # For now, return None to indicate re-planning failed
            # This could be enhanced to actually parse and execute recovery plans
            return None
            
        except Exception as e:
            dprint(f"❌ Failed to create recovery plan: {str(e)}")
            return None
    
    async def _execute_recovery_plan(self, recovery_plan: ExecutionPlan, previous_results: dict) -> Dict[str, Any]:
        """Executes the recovery plan"""
        # This would execute the recovery plan
        # For now, return empty results
        return {}

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
            file_path = kwargs.get('file_path')

            dprint(f"🔍 ScrapingTool debug - kwargs keys: {list(kwargs.keys())}")
            dprint(f"🔍 ScrapingTool debug - url from kwargs: {url}")
            
            # If no URL provided, try to extract from context
            if not url and 'urls' in kwargs:
                url = kwargs['urls'][0] if kwargs['urls'] else None
                dprint(f"🔍 ScrapingTool debug - found url from 'urls': {url}")
            
            # If still no URL, try to extract from query text using regex
            if not url and query:
                dprint("🔍 ScrapingTool: No URL in args, trying to extract from query text...")
                url_match = re.search(r'https?://[^\s,"]+', str(query))
                if url_match:
                    url = url_match.group(0)
                    dprint(f"🔍 ScrapingTool: Extracted URL from query: {url}")

            # If still no URL, try to read from file_path if it exists
            if not url and file_path and os.path.exists(file_path):
                dprint(f"🔍 ScrapingTool: No URL found yet, reading from file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    url_match = re.search(r'https?://[^\s,"]+', file_content)
                    if url_match:
                        url = url_match.group(0)
                        dprint(f"🔍 ScrapingTool: Extracted URL from file content: {url}")
                except Exception as e:
                    dprint(f"⚠️ ScrapingTool: Error reading file {file_path}: {e}")

            if not url:
                dprint(f"❌ ScrapingTool: No URL found in context. Available keys: {list(kwargs.keys())}")
                return "Error: No URL provided for scraping"
            
            dprint(f"🌐 ScrapingTool: Scraping URL '{url}' for query: '{query}'")
            
            from scraper_agent import run_scraping_only_task
            result = await run_scraping_only_task(url, query)
            
            if "error" in result:
                return f"Error scraping website: {result['error']}"
            
            # Store the result in the data cache for other tools to use
            if 'data_cache' in kwargs:
                data_cache = kwargs['data_cache']
                # Store the scraped data with a key that other tools can find
                await data_cache.store('scraped_data', result)
                dprint(f"💾 ScrapingTool: Stored scraped data in cache")
            
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._agent_executed = False

    async def _run(self, **kwargs) -> str:
        """
        Analyzes a DataFrame using LangChain's powerful Pandas Agent.
        """
        import json
        try:
            # Extract parameters from kwargs (context-aware)
            table_id = kwargs.get('table_id')
            query = kwargs.get('query')
            
            # Debug: Print what's available in kwargs
            dprint(f"🔍 DataframeAnalysisTool: Available kwargs keys: {list(kwargs.keys())}")
            dprint(f"🔍 DataframeAnalysisTool: Query received: {query}")
            dprint(f"🔍 DataframeAnalysisTool: Query type: {type(query)}")
            dprint(f"🔍 DataframeAnalysisTool: Query length: {len(str(query)) if query else 0}")
            
            # Preprocess and clean the query using LLM
            if query and isinstance(query, str) and len(query) > 500:  # Only clean if query is long/complex
                dprint(f"🔍 DataframeAnalysisTool: Query is long ({len(query)} chars), cleaning with LLM...")
                try:
                    # Use centralized LLM manager for query cleaning
                    llm_for_cleaning = llm_manager.get_llm(temperature=0.0)
                    
                    # Clean the query using ChatPromptTemplate
                    system_template = """Clean input for data analysis. Keep only the essential information.

KEEP:
- Data schema/columns and their descriptions
- Sample data rows
- The actual questions to be answered
- Data source information (if relevant)

REMOVE:
- File paths and storage structure examples
- SQL examples and code snippets
- Irrelevant metadata
- Duplicate information

Return only the cleaned, focused content."""

                    human_template = """Clean this input for data analysis:

Input to clean:
{query}

Return only the cleaned, focused content:"""

                    # Create the prompt template
                    cleaning_prompt_template = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_template),
                        HumanMessagePromptTemplate.from_template(human_template)
                    ])
                    
                    # Format the prompt
                    formatted_cleaning_prompt = cleaning_prompt_template.format_messages(query=query)
                    
                    # Convert to string format for the LLM
                    cleaning_prompt = ""
                    for message in formatted_cleaning_prompt:
                        if message.type == "system":
                            cleaning_prompt += f"SYSTEM: {message.content}\n\n"
                        elif message.type == "human":
                            cleaning_prompt += f"USER: {message.content}\n\n"

                    cleaned_response = await llm_for_cleaning.ainvoke(cleaning_prompt)
                    if cleaned_response and hasattr(cleaned_response, 'content'):
                        cleaned_query = cleaned_response.content
                        dprint(f"🔍 DataframeAnalysisTool: Query cleaned from {len(query)} to {len(cleaned_query)} chars")
                        query = cleaned_query
                    else:
                        dprint(f"🔍 DataframeAnalysisTool: LLM cleaning failed, using original query")
                except Exception as e:
                    dprint(f"🔍 DataframeAnalysisTool: Error during query cleaning: {e}, using original query")
            
            if 'question_2' in kwargs:
                dprint(f"🔍 DataframeAnalysisTool: question_2 result type: {type(kwargs['question_2'])}")
                dprint(f"🔍 DataframeAnalysisTool: question_2 result preview: {str(kwargs['question_2'])[:200]}...")
            
            # If no table_id provided, try to extract from context or cache
            if not table_id:
                # First try to get from context
                context = None
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
                
                # If still no table_id, check if we have a file_path in context
                if not table_id and 'context' in kwargs:
                    context = kwargs['context']
                    if isinstance(context, dict) and 'file_path' in context:
                        file_path = context['file_path']
                        if file_path and os.path.exists(file_path):
                            # Check if it's a CSV file
                            if file_path.lower().endswith('.csv'):
                                try:
                                    df = pd.read_csv(file_path)
                                    dprint(f"🔍 DataframeAnalysisTool: Successfully loaded CSV from {file_path} with {len(df)} rows")
                                    dprint(f"🔍 DataframeAnalysisTool: DataFrame columns: {list(df.columns)}")
                                    dprint(f"🔍 DataframeAnalysisTool: DataFrame sample:\n{df.head()}")
                                    # Skip the rest of the logic since we have our DataFrame
                                    goto_analysis = True
                                    dprint(f"🔍 DataframeAnalysisTool: Set goto_analysis = True for CSV file")
                                except Exception as e:
                                    dprint(f"🔍 DataframeAnalysisTool: Failed to load CSV from {file_path}: {e}")
                                    goto_analysis = False
                            else:
                                goto_analysis = False
                        else:
                            goto_analysis = False
                    else:
                        goto_analysis = False
                else:
                    goto_analysis = False
            else:
                goto_analysis = False
            
            if goto_analysis:
                # Mark that the first agent is executing
                first_agent_executed = True
                dprint(f"🔍 DataframeAnalysisTool: First agent executing for CSV data")
                
                # Use centralized LLM manager for LangChain
                llm_for_langchain = llm_manager.get_llm(temperature=0.0)

                # Create and run the Pandas DataFrame Agent using ChatPromptTemplate
                # IMPORTANT: Disable SQL generation to prevent JULIANDAY function errors
                
                # Generate data structure example dynamically from actual data FIRST
                try:
                    dprint(f"🔍 DataframeAnalysisTool: Debug - DataFrame shape: {df.shape}")
                    dprint(f"🔍 DataframeAnalysisTool: Debug - DataFrame columns: {list(df.columns)}")
                    dprint(f"🔍 DataframeAnalysisTool: Debug - First row result: {df.iloc[0]['result'][:200]}...")
                    
                    data_structure_example = """# The DataFrame df has this structure:
# df.iloc[0]['result'] contains the first SQL query result
# df.iloc[1]['result'] contains the second SQL query result  
# df.iloc[2]['result'] contains the third SQL query result

# CORRECT PARSING METHOD (SQL results have single quotes, not double quotes):
result_str = df.iloc[0]['result']
if 'Result:' in result_str:
    data_start = result_str.find('[')
    data_end = result_str.rfind(']') + 1
    data_json = result_str[data_start:data_end]
    # Use ast.literal_eval() for SQL results with single quotes
    data = ast.literal_eval(data_json)
    # Now data is a list of dictionaries you can work with"""
                    
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Generated data structure example successfully")
                except Exception as e:
                    dprint(f"❌ DataframeAnalysisTool: Error generating data structure example: {e}")
                    # Fallback to a simple example
                    data_structure_example = """# The DataFrame df contains SQL query results
# Each row has a 'result' column with data in "Result: [...]" format
# Use json.loads() to parse the data after extracting the JSON part"""

                # Create structured prompt template for data analysis - LESS STRICT VERSION
                system_template = """You are a data analysis expert. Use pandas, numpy, matplotlib, seaborn, scipy.stats, re, json, ast.

CRITICAL: DataFrame `df` has REAL data. DO NOT create fake data.

IMPORTS: Start with proper Python syntax:
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                import scipy.stats
                import re
                import json
                import ast
                import io
                import base64

VISUALIZATION - CRITICAL:
❌ NEVER use plt.figure() - this crashes the system
❌ NEVER use plt.gca() - this crashes the system
❌ NEVER use any matplotlib GUI functions
✅ Use seaborn plotting functions ONLY
✅ Save plots to disk under temp_files/ as PNG files
✅ Print a single line with IMAGE_PATH: <path> so the system can inline base64 later

EXAMPLE (save to file and output path only):
import uuid, os
# Create the plot safely with non-interactive matplotlib
fig, ax = plt.subplots(figsize=(8,6))
sns.regplot(x='Year', y='Delay', data=df, ax=ax)
plt.title('Year vs Delay Regression Plot')
os.makedirs('temp_files', exist_ok=True)
image_path = os.path.join('temp_files', "plot_" + str(uuid.uuid4().hex) + ".png")
plt.savefig(image_path, format='png', bbox_inches='tight')
plt.close()
print("IMAGE_PATH: ", image_path)

PANDAS METHODS: Use df.groupby(), df.agg(), df.mean(), df.count(), df.describe(), df.plot(), df.scatter()

STATISTICAL: Use scipy.stats for tests, numpy for operations, sklearn for ML models

CRITICAL: You MUST answer ALL questions completely. Do not stop until all questions are answered.

FINAL ANSWER: After answering all questions, include a single line with IMAGE_PATH: <path> for each plot you saved.

CODE EXECUTION RULES:
- NEVER execute incomplete code
- Wait for complete code blocks before running
- Validate syntax before execution
- Execute only when you have a complete, runnable program

CODE QUALITY - CRITICAL:
✅ ALWAYS review your code before returning it
✅ Check for balanced parentheses (), brackets [], and quotes ''
✅ Ensure Python syntax is valid
✅ Test code structure mentally before execution
✅ If you find syntax errors, fix them before returning

DATA STRUCTURE EXAMPLE - CRITICAL:
{data_structure_example}

WORKING EXAMPLES:
# Use pandas methods to analyze the data
# Create visualizations when requested
# Handle data parsing intelligently based on the actual data structure

**CRITICAL: ALWAYS validate Python syntax before returning code**
**CRITICAL: Check bracket matching: every square and curly brackets must be balanced**
**CRITICAL: Use proper Python dictionary/list syntax with balanced brackets and quotes**
**CRITICAL: Test your code structure mentally before returning**

NEVER use: data[0]['regr_slope(...)'] or hardcoded assumptions"""

                # Create the prompt template
                analysis_prompt_template = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_template)
                ])
                
                # Format the prompt with the data structure example
                formatted_prefix = analysis_prompt_template.format_messages(
                    data_structure_example=data_structure_example
                )
                prefix_text = formatted_prefix[0].content
                
                dprint(f"🔍 DataframeAnalysisTool: Debug - Final prompt length: {len(prefix_text)}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Prompt preview: {prefix_text[:300]}...")
                
                try: 
                    pandas_agent = create_pandas_dataframe_agent(
                        llm_for_langchain, 
                        df, 
                        prefix=prefix_text,
                        suffix=None,  # Add this for proper prompt control
                        verbose=True,
                        allow_dangerous_code=True, # Opt-in to code execution
                        max_iterations=5,
                        max_execution_time = 300,  # Explicit timeout in executor kwargs
                        early_stopping_method="force",
                        agent_type="zero-shot-react-description",  # Use the most stable agent type
                        return_intermediate_steps=True,
                        agent_executor_kwargs={
                            "handle_parsing_errors": True,
                        }
                    )
                
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Agent created successfully")
                    dprint(f"🔍 DataframeAnalysisTool: Debug - About to invoke agent with query: {query[:100]}...")
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Agent type: {type(pandas_agent)}")
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Agent attributes: {dir(pandas_agent)}")
                except Exception as creation_error:
                    dprint(f"❌ DataframeAnalysisTool: Agent creation failed: {creation_error}")
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Creation error type: {type(creation_error)}")
                    import traceback
                    traceback.print_exc()
                    return json.dumps({"message": f"Failed to create pandas agent: {str(creation_error)}"})
                # Use a more robust timeout mechanism that actually works with LangChain agents
                async def execute_agent_with_timeout():
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Starting agent execution with robust timeout...")
                    start_time = asyncio.get_event_loop().time()
                    
                    # Create a task for the agent execution
                    agent_task = asyncio.create_task(pandas_agent.ainvoke(query))
                    
                    try:
                        # Wait for the task with timeout
                        result = await asyncio.wait_for(agent_task, timeout=300)
                        end_time = asyncio.get_event_loop().time()
                        dprint(f"🔍 DataframeAnalysisTool: Debug - Agent execution completed in {end_time - start_time:.2f} seconds")
                        return result
                    except asyncio.TimeoutError:
                        dprint(f"🔍 DataframeAnalysisTool: Debug - Timeout reached, cancelling agent task...")
                        # Cancel the agent task
                        agent_task.cancel()
                        try:
                            # Wait a bit for the task to cancel
                            await asyncio.wait_for(agent_task, timeout=5)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                        raise asyncio.TimeoutError("Agent execution timed out after 5 minutes")
                    except Exception as e:
                        end_time = asyncio.get_event_loop().time()
                        dprint(f"🔍 DataframeAnalysisTool: Debug - Agent execution failed after {end_time - start_time:.2f} seconds with error: {e}")
                        # Cancel the task if it's still running
                        if not agent_task.done():
                            agent_task.cancel()
                        raise
                
                try:
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Starting robust timeout wrapper (300s)...")
                    response = await execute_agent_with_timeout()
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Robust timeout wrapper completed successfully")
                except asyncio.TimeoutError:
                    dprint(f"❌ DataframeAnalysisTool: Agent execution timed out after 5 minutes")
                    return json.dumps({"message": "Agent execution timed out after 5 minutes. Please try a simpler query."})
                except Exception as e:
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Agent execution failed with exception: {e}")
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    return json.dumps({"message": f"Agent execution failed: {str(e)}"})
                
                dprint(f"🔍 DataframeAnalysisTool: Debug - Agent response type: {type(response)}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Agent response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")

                # Prefer deterministic JSON emitted by tool code over LLM summary
                def _extract_last_json_string(_resp: dict) -> str | None:
                    import json as _json
                    try:
                        steps = list(_resp.get("intermediate_steps", []))
                    except Exception:
                        steps = []
                    for item in reversed(steps):
                        try:
                            _action, obs = item
                        except Exception:
                            continue
                        if isinstance(obs, str):
                            s = obs.strip()
                            if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                                try:
                                    _json.loads(s)
                                    return s
                                except Exception:
                                    pass
                    out = _resp.get("output", "") if isinstance(_resp, dict) else ""
                    if isinstance(out, str):
                        s = out.strip()
                        if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                            try:
                                _json.loads(s)
                                return s
                            except Exception:
                                pass
                    return None

                json_block = _extract_last_json_string(response) if isinstance(response, dict) else None
                answer = json_block if json_block is not None else (response.get("output", "Analysis completed, but no specific output was generated.") if isinstance(response, dict) else "Analysis completed, but no specific output was generated.")

                dprint(f"🔍 DataframeAnalysisTool: Debug - Deterministic answer: {str(answer)[:500]}...")
                
                # Note: IMAGE_PATH -> base64 conversion now happens in ResponseFormatter
                
                # CRITICAL: Ensure base64 images are returned completely
                # Check if the answer contains a base64 image that might be truncated
                if "base64" in answer.lower() and len(answer) > 1000:
                    dprint(f"🔍 DataframeAnalysisTool: Detected base64 image in answer, length: {len(answer)}")
                    # Ensure we return the complete answer with the full base64 string
                    # CRITICAL: Mark this tool instance as having executed an agent
                    self._agent_executed = True
                    return answer
                
                # Get the agent's output directly - no more code extraction
                # Base64 encoded plots are included directly in the agent's output
                
                # CRITICAL: Mark this tool instance as having executed an agent
                self._agent_executed = True
                return answer

            # CRITICAL FIX: If we already processed data (goto_analysis was True), don't continue to the second agent
            # This prevents duplicate execution of LangChain agents
            if 'df' in locals() and df is not None and goto_analysis:
                dprint(f"🔍 DataframeAnalysisTool: Data already processed by first agent, skipping second execution")
                return "Data analysis completed by first agent"

            # CRITICAL FIX: If the first agent already executed, don't run the second one
            if 'first_agent_executed' in locals() and first_agent_executed:
                dprint(f"🔍 DataframeAnalysisTool: First agent already executed, skipping second execution to prevent duplication")
                return "Data analysis completed by first agent - duplicate execution prevented"

            # CRITICAL FIX: If we have a successful result from the first agent, return it immediately
            # This prevents the second agent from overwriting successful results
            if 'answer' in locals() and answer and len(answer) > 100:
                dprint(f"🔍 DataframeAnalysisTool: First agent completed successfully, returning result immediately")
                return answer

            # Check if we have SQL analysis results in the cache
            sql_result_data = None
            if 'data_cache' in kwargs:
                try:
                    sql_result_data = await kwargs['data_cache'].retrieve("latest_sql_result")
                    dprint(f"🔍 DataframeAnalysisTool: Found data in cache: {type(sql_result_data)}")
                except Exception as e:
                    dprint(f"🔍 DataframeAnalysisTool: Cache retrieval failed: {e}")
            
            # CRITICAL FIX: Look for SQL results from previous stage tasks
            if sql_result_data is None:
                # Look for any task result from Stage 1 that might contain data
                for task_key in kwargs.keys():
                    if task_key not in ['query', 'file_path', 'data_cache', 'data_source_type', 'user_query', 'urls', 'database_info', 'file_info']:
                        task_result = kwargs[task_key]
                        dprint(f"🔍 DataframeAnalysisTool: Checking {task_key}: {type(task_result)}")
                        dprint(f"🔍 DataframeAnalysisTool: {task_key} content preview: {str(task_result)[:300]}...")
                        
                        if isinstance(task_result, str) and 'Result:' in task_result:
                            dprint(f"🔍 DataframeAnalysisTool: Found task result with 'Result:' pattern")
                            dprint(f"🔍 DataframeAnalysisTool: Full task result preview: {task_result[:500]}...")
                            # Extract the result data from the string
                            try:
                                # Parse the result string to extract the actual data
                                import re
                                import ast
                                
                                # Look for the result pattern: "Result: [{...}, {...}]"
                                result_match = re.search(r'Result: (\[.*?\])', task_result)
                                if result_match:
                                    result_str = result_match.group(1)
                                    dprint(f"🔍 DataframeAnalysisTool: Extracted result string: {result_str[:200]}...")
                                    try:
                                        # Convert string representation back to list of dicts
                                        sql_result_data = ast.literal_eval(result_str)
                                        dprint(f"🔍 DataframeAnalysisTool: Successfully parsed data with {len(sql_result_data)} rows")
                                        break
                                    except Exception as parse_error:
                                        dprint(f"🔍 DataframeAnalysisTool: Failed to parse extracted result: {parse_error}")
                                        dprint(f"🔍 DataframeAnalysisTool: Raw extracted string: {result_str}")
                                        # Try alternative parsing method
                                        try:
                                            # Look for any JSON array in the result
                                            json_match = re.search(r'\[.*?\]', result_str)
                                            if json_match:
                                                alt_result_str = json_match.group(0)
                                                dprint(f"🔍 DataframeAnalysisTool: Trying alternative parsing with: {alt_result_str[:200]}...")
                                                sql_result_data = ast.literal_eval(alt_result_str)
                                                dprint(f"🔍 DataframeAnalysisTool: Alternative parsing successful with {len(sql_result_data)} rows")
                                                break
                                        except Exception as alt_error:
                                            dprint(f"🔍 DataframeAnalysisTool: Alternative parsing also failed: {alt_error}")
                                        continue
                            except Exception as e:
                                dprint(f"🔍 DataframeAnalysisTool: Failed to parse {task_key} result: {e}")
                                continue
            
            # Initialize df variable and execution tracking
            df = None
            first_agent_executed = False
            goto_analysis = False  # Initialize goto_analysis variable
            
            # CRITICAL: Global execution control - prevent multiple agent runs
            if hasattr(self, '_agent_executed') and self._agent_executed:
                dprint(f"🔍 DataframeAnalysisTool: Agent already executed in this tool instance, skipping")
                return "Agent already executed - duplicate execution prevented"
            
            # DISABLED: Check if we've already analyzed this data to prevent duplicate runs
            # The issue is that different tasks are getting different results, causing cache corruption
            # analysis_cache_key = f"analysis_completed_{hash(str(sql_result_data))}_{hash(query)}"
            # if 'data_cache' in kwargs:
            #     try:
            #         already_analyzed = await kwargs['data_cache'].retrieve(analysis_cache_key)
            #         if already_analyzed:
            #             print(f"🔍 DataframeAnalysisTool: Analysis already completed for this data, returning cached result")
            #             return already_analyzed
            #     except Exception as e:
            #         print(f"🔍 DataframeAnalysisTool: Cache check failed: {e}")
            
            if sql_result_data is not None:
                # Use the SQL result data for analysis/plotting
                # Extract the actual data content from SQL results, not the metadata
                dprint(f"🔍 DataframeAnalysisTool: SQL result data structure: {type(sql_result_data)}")
                dprint(f"🔍 DataframeAnalysisTool: SQL result data preview: {str(sql_result_data)[:500]}...")
                
                # Look for the actual data content in the SQL results
                actual_data = None
                for item in sql_result_data:
                    if isinstance(item, dict) and 'result' in item:
                        result_str = item['result']
                        dprint(f"🔍 DataframeAnalysisTool: Processing result: {result_str[:200]}...")
                        
                        if 'Result:' in result_str:
                            # Extract the data after "Result:"
                            try:
                                import re
                                import ast
                                
                                # Look for the result pattern: "Result: [{...}, {...}]"
                                # Use greedy matching to get the complete data structure
                                result_match = re.search(r'Result: (\[.*\])', result_str, re.DOTALL)
                                dprint(f"🔍 DataframeAnalysisTool: Regex match result: {result_match}")
                                
                                if result_match:
                                    result_str = result_match.group(1)
                                    dprint(f"🔍 DataframeAnalysisTool: Extracted result string: {result_str[:200]}...")
                                    try:
                                        # Convert string representation back to list of dicts
                                        actual_data = ast.literal_eval(result_str)
                                        dprint(f"🔍 DataframeAnalysisTool: Successfully parsed actual data with {len(actual_data)} rows")
                                        break
                                    except Exception as parse_error:
                                        dprint(f"🔍 DataframeAnalysisTool: Failed to parse extracted result: {parse_error}")
                                        dprint(f"🔍 DataframeAnalysisTool: Raw extracted string: {result_str}")
                                        # Try alternative parsing method
                                        try:
                                            # Look for any JSON array in the result with greedy matching
                                            json_match = re.search(r'\[.*\]', result_str, re.DOTALL)
                                            if json_match:
                                                alt_result_str = json_match.group(0)
                                                dprint(f"🔍 DataframeAnalysisTool: Trying alternative parsing with: {alt_result_str[:200]}...")
                                                actual_data = ast.literal_eval(alt_result_str)
                                                dprint(f"🔍 DataframeAnalysisTool: Alternative parsing successful with {len(actual_data)} rows")
                                                break
                                        except Exception as alt_error:
                                            dprint(f"🔍 DataframeAnalysisTool: Alternative parsing also failed: {alt_error}")
                                        continue
                                else:
                                    dprint(f"🔍 DataframeAnalysisTool: No regex match found for pattern 'Result: (\\[.*\\])'")
                                    dprint(f"🔍 DataframeAnalysisTool: Full result string: {result_str}")
                            except Exception as e:
                                dprint(f"🔍 DataframeAnalysisTool: Error extracting data: {e}")
                                continue
                
                if actual_data:
                    # Use the actual data content
                    df = pd.DataFrame(actual_data)
                    dprint(f"🔍 DataframeAnalysisTool: Using ACTUAL SQL data with {len(df)} rows")
                    dprint(f"🔍 DataframeAnalysisTool: DataFrame columns: {list(df.columns)}")
                    dprint(f"🔍 DataframeAnalysisTool: DataFrame sample:\n{df.head()}")
                else:
                    # Fallback: use the SQL result structure but warn about it
                    dprint(f"⚠️ DataframeAnalysisTool: Could not extract actual data, using SQL result structure")
                    df = pd.DataFrame(sql_result_data)
                    dprint(f"🔍 DataframeAnalysisTool: Using SQL result structure with {len(df)} rows")
                    dprint(f"🔍 DataframeAnalysisTool: DataFrame columns: {list(df.columns)}")
                    dprint(f"🔍 DataframeAnalysisTool: DataFrame sample:\n{df.head()}")
                
                # Don't clear the cache immediately - let it persist for this session
                # Only clear if we're done with all operations
                # if 'data_cache' in kwargs:
                #     await kwargs['data_cache'].store("latest_sql_result", None)
            elif table_id:
                    # Use the table file as before
                table_path = f"temp_files/{table_id}"
                if not os.path.exists(table_path):
                        return json.dumps({"message": f"Table file not found: {table_path}"})
                
                df = pd.read_csv(table_path)
            elif 'context' in kwargs:
                context = kwargs['context']
                if isinstance(context, dict) and 'file_path' in context:
                    file_path = context['file_path']
                    if file_path and os.path.exists(file_path):
                        # Check if it's a CSV file
                        if file_path.lower().endswith('.csv'):
                            try:
                                df = pd.read_csv(file_path)
                                dprint(f"🔍 DataframeAnalysisTool: Successfully loaded CSV from {file_path} with {len(df)} rows")
                                dprint(f"🔍 DataframeAnalysisTool: DataFrame columns: {list(df.columns)}")
                                dprint(f"🔍 DataframeAnalysisTool: DataFrame sample:\n{df.head()}")
                            except Exception as e:
                                dprint(f"🔍 DataframeAnalysisTool: Failed to load CSV from {file_path}: {e}")
                                return json.dumps({"message": f"Failed to load CSV file: {str(e)}"})
                        else:
                            # For non-CSV files (like .txt), don't try to load as data
                            # The file contains the query/description, not the actual data
                            dprint(f"🔍 DataframeAnalysisTool: File {file_path} is not CSV, treating as query content only")
                            # Continue without loading file as data - use whatever data is available from other sources
                    else:
                        return json.dumps({"message": f"File not found: {file_path}"})
            elif 'file_path' in kwargs:
                # Check if file_path is directly in kwargs
                file_path = kwargs['file_path']
                if file_path and os.path.exists(file_path):
                    # Check if it's a CSV file
                    if file_path.lower().endswith('.csv'):
                        try:
                            df = pd.read_csv(file_path)
                            dprint(f"🔍 DataframeAnalysisTool: Successfully loaded CSV from {file_path} with {len(df)} rows")
                            dprint(f"🔍 DataframeAnalysisTool: DataFrame columns: {list(df.columns)}")
                            dprint(f"🔍 DataframeAnalysisTool: DataFrame sample:\n{df.head()}")
                            # Set goto_analysis to True for CSV files to prevent second agent execution
                            goto_analysis = True
                            dprint(f"🔍 DataframeAnalysisTool: Set goto_analysis = True for CSV file from kwargs")
                        except Exception as e:
                            dprint(f"🔍 DataframeAnalysisTool: Failed to load CSV from {file_path}: {e}")
                            return json.dumps({"message": f"Failed to load CSV file: {str(e)}"})
                    else:
                        # For non-CSV files (like .txt), don't try to load as data
                        # The file contains the query/description, not the actual data
                        dprint(f"🔍 DataframeAnalysisTool: File {file_path} is not CSV, treating as query content only")
                        # Continue without loading file as data - use whatever data is available from other sources
                else:
                    return json.dumps({"message": f"File not found: {file_path}"})
            
            # CRITICAL FIX: If the first agent already executed, don't run the second one
            if 'first_agent_executed' in locals() and first_agent_executed:
                dprint(f"🔍 DataframeAnalysisTool: First agent already executed, skipping second execution to prevent duplication")
                return "Data analysis completed by first agent - duplicate execution prevented"

            # CRITICAL FIX: Check for data from previous tasks (like get_data_for_analysis)
            if df is None:
                # Look for results from previous tasks that might contain data
                for key, value in kwargs.items():
                    if key != 'query' and key != 'file_path' and key != 'data_cache':
                        dprint(f"🔍 DataframeAnalysisTool: Checking previous task result: {key}")
                        dprint(f"🔍 DataframeAnalysisTool: Value type: {type(value)}")
                        dprint(f"🔍 DataframeAnalysisTool: Value content: {repr(value)[:200]}")
                        
                        if isinstance(value, str) and len(value) > 100:
                            # This might be data from a previous task
                            dprint(f"🔍 DataframeAnalysisTool: Found potential data from {key}, length: {len(value)}")
                            # Try to parse as JSON or use as raw data
                            try:
                                # Check if it's JSON data
                                if value.strip().startswith('[') or value.strip().startswith('{'):
                                    import json
                                    parsed_data = json.loads(value)
                                    if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                        df = pd.DataFrame(parsed_data)
                                        dprint(f"🔍 DataframeAnalysisTool: Successfully parsed data from {key} into DataFrame with {len(df)} rows")
                                        break
                                    elif isinstance(parsed_data, dict) and 'data' in parsed_data:
                                        df = pd.DataFrame(parsed_data['data'])
                                        dprint(f"🔍 DataframeAnalysisTool: Successfully parsed data from {key} into DataFrame with {len(df)} rows")
                                        break
                            except Exception as e:
                                dprint(f"🔍 DataframeAnalysisTool: Failed to parse data from {key}: {e}")
                                continue
                        elif value is None:
                            dprint(f"🔍 DataframeAnalysisTool: {key} is None - task completed but returned no data")
                        elif isinstance(value, (list, dict)):
                            dprint(f"🔍 DataframeAnalysisTool: {key} is {type(value)} with content: {repr(value)[:200]}")
                
                # If still no DataFrame, return error with detailed debugging info
                if df is None:
                    debug_info = f"""
                    Error: No data provided for analysis. 
                    
                    Available context keys: {list(kwargs.keys())}
                    Data cache keys: {kwargs.get('data_cache', {}).get_all_keys() if hasattr(kwargs.get('data_cache', {}), 'get_all_keys') else 'No cache'}
                    
                    Previous task results checked:
                    - data_source_type: {kwargs.get('data_source_type', 'Not found')}
                    - database_info: {kwargs.get('database_info', 'Not found')}
                    
                    Please ensure the data acquisition task completed successfully before running analysis.
                    """
                    dprint(f"🔍 DataframeAnalysisTool: {debug_info}")
                    return debug_info

            # --- Data Validation Guardrail ---
            
            if df.empty:
                return json.dumps({"answer": "The data provided is empty. I cannot perform any analysis."})
            
            # For plotting, we need at least 2 data points
            if "plot" in query.lower() or "scatter" in query.lower() or "graph" in query.lower():
                if len(df) < 2:
                    return json.dumps({
                        "answer": f"I cannot create a plot because there is not enough data. The dataset only contains {len(df)} row(s). Meaningful plotting requires at least two data points."
                    })
            # ---------------------------------

            # CRITICAL: Check if we already have a successful result before creating the second agent
            if 'answer' in locals() and answer and len(answer) > 100:
                dprint(f"🔍 DataframeAnalysisTool: First agent already completed successfully, skipping second agent creation")
                return answer
            
            # SECOND AGENT: This only runs when the first agent didn't execute (e.g., for SQL results, not CSV files)
            dprint(f"🔍 DataframeAnalysisTool: Second agent executing for non-CSV data")
            
            # Use centralized LLM manager for LangChain
            llm_for_langchain = llm_manager.get_llm(temperature=0.0)

            # Create and run the Pandas DataFrame Agent using ChatPromptTemplate
            # IMPORTANT: Disable SQL generation to prevent JULIANDAY function errors
            
            # Generate data structure example dynamically from actual data FIRST
            try:
                dprint(f"🔍 DataframeAnalysisTool: Debug - DataFrame shape: {df.shape}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - DataFrame columns: {list(df.columns)}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - First row result: {df.iloc[0]['result'][:200]}...")
                
                data_structure_example = """# The DataFrame df has this structure:
# df.iloc[0]['result'] contains the first SQL query result
# df.iloc[1]['result'] contains the second SQL query result  
# df.iloc[2]['result'] contains the third SQL query result

# CORRECT PARSING METHOD (SQL results have single quotes, not double quotes):
result_str = df.iloc[0]['result']
if 'Result:' in result_str:
    data_start = result_str.find('[')
    data_end = result_str.rfind(']') + 1
    data_json = result_str[data_start:data_end]
    # Use ast.literal_eval() for SQL results with single quotes
    data = ast.literal_eval(data_json)
    # Now data is a list of dictionaries you can work with"""
                
                dprint(f"🔍 DataframeAnalysisTool: Debug - Generated data structure example successfully")
            except Exception as e:
                dprint(f"❌ DataframeAnalysisTool: Error generating data structure example: {e}")
                # Fallback to a simple example
                data_structure_example = """# The DataFrame df contains SQL query results
# Each row has a 'result' column with data in "Result: [...]" format
# Use json.loads() to parse the data after extracting the JSON part"""

            # Create structured prompt template for advanced data analysis - LESS STRICT VERSION
            system_template = """You are a data analysis expert. Use pandas, numpy, matplotlib, seaborn, scipy.stats, re, json, ast. Follow ReAct format: Thought → Action → Action Input.

CRITICAL: DataFrame `df` has REAL data. DO NOT create fake data.

IMPORTS: Start with proper Python syntax:
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                import scipy.stats
                import re
                import json
                import ast
                import io
                import base64

VISUALIZATION - CRITICAL:
❌ NEVER use plt.figure() - this crashes the system
❌ NEVER use plt.gca() - this crashes the system
❌ NEVER use any matplotlib GUI functions
✅ Use seaborn plotting functions ONLY
✅ Save plots to disk under temp_files/ as PNG files
✅ Print a single line with IMAGE_PATH: <path> so the system can inline base64 later

EXAMPLE (save to file and output path only):
import uuid, os
# Create the plot safely with non-interactive matplotlib
fig, ax = plt.subplots(figsize=(8,6))
sns.regplot(x='Year', y='Delay', data=df, ax=ax)
plt.title('Year vs Delay Regression Plot')
os.makedirs('temp_files', exist_ok=True)
image_path = os.path.join('temp_files', "plot_" + str(uuid.uuid4().hex) + ".png")
plt.savefig(image_path, format='png', bbox_inches='tight')
plt.close()
print("IMAGE_PATH: ", image_path)

MULTI-QUESTION: Answer each numbered question individually. Use semantic reasoning to map columns to questions.

SEMANTIC REASONING: Don't just look for exact column name matches. Think about what each column conceptually represents. If a question asks for X but you don't have X exactly, think: "What do I have that could answer this?"

DATA STRUCTURE: DataFrame has ONE ROW per SQL query result. Access safely with df.iloc[0]['result'], df.iloc[1]['result'], etc.

CRITICAL: You MUST answer ALL questions completely. Do not stop until all questions are answered.

FINAL ANSWER: After answering all questions, include a single line with IMAGE_PATH: <path> for each plot you saved.

CODE EXECUTION RULES:
- NEVER execute incomplete code
- Wait for complete code blocks before running
- Validate syntax before execution
- Execute only when you have a complete, runnable program

CODE QUALITY - CRITICAL:
✅ ALWAYS review your code before returning it
✅ Check for balanced parentheses (), brackets [], and quotes ''
✅ Ensure Python syntax is valid
✅ Test code structure mentally before execution
✅ If you find syntax errors, fix them before returning

PYTHON CODE GENERATION - CRITICAL:
✅ Each statement must be on its own line
✅ Each print statement must be complete and properly formatted
✅ No missing newlines between statements
✅ No incomplete or truncated code
✅ All code blocks must be syntactically complete
✅ Test your code mentally before returning it

NEVER return incomplete or malformed code!

DATA STRUCTURE EXAMPLE - CRITICAL:
{data_structure_example}

WORKING EXAMPLES:
# Use pandas methods to analyze the data
# Create visualizations when requested
# Handle data parsing intelligently based on the actual data structure

**CRITICAL: ALWAYS validate Python syntax before returning code**
**CRITICAL: Check bracket matching: every square and curly brackets must be balanced**
**CRITICAL: Use proper Python dictionary/list syntax with balanced brackets and quotes**
**CRITICAL: Test your code structure mentally before returning**

NEVER use: data[0]['regr_slope(...)'] or hardcoded assumptions"""

            # Create the prompt template
            advanced_analysis_prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template)
            ])
            
            # Format the prompt with the data structure example
            formatted_prefix = advanced_analysis_prompt_template.format_messages(
                data_structure_example=data_structure_example
            )
            prefix_text = formatted_prefix[0].content
            
            dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent prompt length: {len(prefix_text)}")
            dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent prompt preview: {prefix_text[:300]}...")
            
            try:
                pandas_agent = create_pandas_dataframe_agent(
                    llm_for_langchain, 
                    df, 
                    prefix=prefix_text,
                    suffix=None,  # Add this for proper prompt control
                    verbose=True,
                    allow_dangerous_code=True, # Opt-in to code execution
                    max_iterations=5,
                    early_stopping_method="force",
                    max_execution_time = 300,  # Explicit timeout in executor kwargs
                    agent_type="zero-shot-react-description",  # Use the most stable agent type
                    return_intermediate_steps=True,
                    agent_executor_kwargs={
                        "handle_parsing_errors": True,
                    }
                )
                
                dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent created successfully")
                dprint(f"🔍 DataframeAnalysisTool: Debug - About to invoke second agent with query: {query[:100]}...")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent type: {type(pandas_agent)}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent attributes: {dir(pandas_agent)}")
            except Exception as creation_error:
                dprint(f"❌ DataframeAnalysisTool: Agent creation failed: {creation_error}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Creation error type: {type(creation_error)}")
                import traceback
                traceback.print_exc()
                return json.dumps({"message": f"Failed to create pandas agent: {str(creation_error)}"})
            # Use a more robust timeout mechanism that actually works with LangChain agents
            async def execute_agent_with_timeout():
                dprint(f"🔍 DataframeAnalysisTool: Debug - Starting second agent execution with robust timeout...")
                start_time = asyncio.get_event_loop().time()
                
                # Create a task for the agent execution
                agent_task = asyncio.create_task(pandas_agent.ainvoke(query))
                
                try:
                    # Wait for the task with timeout
                    result = await asyncio.wait_for(agent_task, timeout=300)
                    end_time = asyncio.get_event_loop().time()
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent execution completed in {end_time - start_time:.2f} seconds")
                    return result
                except asyncio.TimeoutError:
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Timeout reached, cancelling second agent task...")
                    # Cancel the agent task
                    agent_task.cancel()
                    try:
                        # Wait a bit for the task to cancel
                        await asyncio.wait_for(agent_task, timeout=5)
                        await agent_task
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    raise asyncio.TimeoutError("Second agent execution timed out after 5 minutes")
                except Exception as e:
                    end_time = asyncio.get_event_loop().time()
                    dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent execution failed after {end_time - start_time:.2f} seconds with error: {e}")
                    # Cancel the task if it's still running
                    if not agent_task.done():
                        agent_task.cancel()
                    raise
            
            try:
                dprint(f"🔍 DataframeAnalysisTool: Debug - Starting robust timeout wrapper for second agent (300s)...")
                response = await execute_agent_with_timeout()
                dprint(f"🔍 DataframeAnalysisTool: Debug - Robust timeout wrapper for second agent completed successfully")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent response type: {type(response)}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")

                # Prefer deterministic tool observations over the LLM final summary (second agent)
                observations2: list[str] = []
                try:
                    if isinstance(response, dict) and "intermediate_steps" in response:
                        for _action, obs in response["intermediate_steps"]:
                            if isinstance(obs, str):
                                observations2.append(obs)
                except Exception as _:
                    pass
                raw_output2 = response.get("output", "") if isinstance(response, dict) else ""
                joined2 = "\n".join([*observations2, raw_output2]).strip()
                answer = joined2 or "Analysis completed, but no specific output was generated."
                dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent deterministic answer: {answer[:500]}...")
                
                # Note: IMAGE_PATH -> base64 conversion now happens in ResponseFormatter
                
            except asyncio.TimeoutError:
                dprint(f"❌ DataframeAnalysisTool: Second agent execution timed out after 5 minutes")
                return json.dumps({"message": "Second agent execution timed out after 5 minutes. Please try a simpler query."})

            except Exception as agent_error:
                dprint(f"🔍 DataframeAnalysisTool: Second agent execution failed: {agent_error}")
                dprint(f"🔍 DataframeAnalysisTool: Debug - Second agent exception type: {type(agent_error)}")
                import traceback
                traceback.print_exc()
                # Initialize answer variable even when agent fails
                answer = f"Agent execution failed: {agent_error}"
            
            # CRITICAL: Ensure base64 images are returned completely
            # Check if the answer contains a base64 image that might be truncated
            if "base64" in answer.lower() and len(answer) > 1000:
                dprint(f"🔍 DataframeAnalysisTool: Second agent detected base64 image in answer, length: {len(answer)}")
                # Ensure we return the complete answer with the full base64 string
                # CRITICAL: Mark this tool instance as having executed an agent
                self._agent_executed = True
                return answer
            
            # Get the agent's output directly - no more code extraction
            # Base64 encoded plots are included directly in the agent's output
            
            # CRITICAL: Mark this tool instance as having executed an agent
            self._agent_executed = True
            return answer

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Ensure json is available in this scope
            import json
            return json.dumps({"message": f"LangChain agent failed: {str(e)}"})

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
            
            # Also accept top-level file_path from executor context
            if not pdf_path and 'file_path' in kwargs:
                pdf_path = kwargs['file_path']
            
            # If no pdf_path provided, try to extract from context
            if not pdf_path and 'context' in kwargs:
                context = kwargs['context']
                if isinstance(context, dict) and 'file_path' in context:
                    pdf_path = context['file_path']
            
            if not pdf_path:
                return "Error: No PDF path provided for analysis"
            
            dprint(f"📄 PDFAnalysisTool: Analyzing PDF '{pdf_path}' with query: '{query}'")
            
            # Call the existing PDF analysis function
            from pdf_parser import run_pdf_analysis_task
            result = await run_pdf_analysis_task(pdf_path, query)
            
            if "error" in result:
                return json.dumps({"message": result["error"]})
            
            # Extract the answer from the search analysis
            search_analysis = result.get("search_analysis", {})
            
            # Handle the new response structure from search_and_analyze
            if isinstance(search_analysis, dict) and "analysis" in search_analysis:
                # New structure: search_analysis.analysis.answer
                answer = search_analysis.get("analysis", {}).get("answer", "No answer found in the PDF.")
                confidence = search_analysis.get("analysis", {}).get("confidence", "unknown")
                method = search_analysis.get("method", "unknown")
                
                return json.dumps({
                     
                    "answer": answer,
                    "confidence": confidence,
                    "method": method
                })
            elif isinstance(search_analysis, dict) and "answer" in search_analysis:
                # Legacy structure: search_analysis.answer
                answer = search_analysis.get("answer", "No answer found in the PDF.")
                return json.dumps({ "answer": answer})
            else:
                # Fallback: return the raw search_analysis
                return json.dumps({ "answer": str(search_analysis)})
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return json.dumps({"message": f"PDF analysis failed: {str(e)}"})

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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dprint(f"🔍 SQLAnalysisTool: Initialized with name: {self.name}")
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            # Use centralized LLM manager
            llm = llm_manager.get_llm(temperature=0.0)
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"
    
    def _adapt_sql_for_database(self, sql_query: str, database_type: str) -> str:
        """Use the global SQL mapper for database-specific adaptations"""
        return sql_mapper.adapt_sql_for_database(sql_query, database_type)
    
    def _contains_dangerous_operations(self, query: str) -> bool:
        """Check if query contains dangerous SQL operations"""
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER', 'TRUNCATE',
            'EXEC', 'EXECUTE', 'EXECUTE IMMEDIATE', 'EXECUTE PROCEDURE'
        ]
        
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return True
        return False
    
    def _validate_basic_syntax(self, query: str) -> bool:
        """Basic SQL syntax validation - more lenient for LLM-generated queries"""
        try:
            # Clean the query first
            clean_query = query.strip()
            
            # Must start with SELECT
            if not clean_query.upper().startswith('SELECT'):
                dprint(f"🔍 SQLValidation: Query doesn't start with SELECT: {clean_query[:50]}...")
                return False
            
            # Check for balanced parentheses (allow slight imbalance for complex queries)
            open_parens = clean_query.count('(')
            close_parens = clean_query.count(')')
            if abs(open_parens - close_parens) > 1:  # Allow 1 imbalance
                dprint(f"🔍 SQLValidation: Significant parenthesis imbalance: {open_parens} open, {close_parens} close")
                return False
            
            # Check for balanced quotes (allow slight imbalance for complex queries)
            single_quotes = clean_query.count("'")
            double_quotes = clean_query.count('"')
            if single_quotes % 2 != 0 and single_quotes > 0:
                dprint(f"🔍 SQLValidation: Unbalanced single quotes: {single_quotes}")
                return False
            if double_quotes % 2 != 0 and double_quotes > 0:
                dprint(f"🔍 SQLValidation: Unbalanced double quotes: {double_quotes}")
                return False
            
            # Basic length check
            if len(clean_query) < 10:
                dprint(f"🔍 SQLValidation: Query too short: {len(clean_query)} chars")
                return False
            
            dprint(f"🔍 SQLValidation: Query passed basic syntax validation")
            return True
            
        except Exception as e:
            dprint(f"🔍 SQLValidation: Exception during validation: {e}")
            return False

    async def _run(self, **kwargs):
        dprint(f"🔍 SQLAnalysisTool: _run called with kwargs keys: {list(kwargs.keys())}")
        
        # Validate and extract required parameters using SQLInput schema
        try:
        # Extract parameters from kwargs (context-aware)
            query = kwargs.get('query')
            if not query:
                return "Error: 'query' parameter is required"
            
            database_type = kwargs.get('database_type', 'duckdb')
            connection_string = kwargs.get('connection_string')
            table_info = kwargs.get('table_info')
                
            # Validate database_type
            valid_db_types = ['duckdb', 'postgresql', 'mysql', 'sqlite']
            if database_type.lower() not in valid_db_types:
                return f"Error: Invalid database_type '{database_type}'. Supported types: {', '.join(valid_db_types)}"
            
            dprint(f"🔍 SQLAnalysisTool: query = {repr(query)}")
            dprint(f"🔍 SQLAnalysisTool: database_type = {database_type}")
            dprint(f"🔍 SQLAnalysisTool: connection_string = {repr(connection_string)}")
            
        except Exception as e:
            return f"Error: Parameter validation failed: {str(e)}"
        
        # If no connection_string provided, try to extract from database_info directly in kwargs
        if not connection_string and 'database_info' in kwargs:
            db_info = kwargs['database_info']
            dprint(f"🔍 SQLAnalysisTool: Found database_info: {db_info}")
            if isinstance(db_info, dict):
                connection_string = db_info.get('connection')
                database_type = db_info.get('type', database_type)
                dprint(f"🔍 SQLAnalysisTool: Extracted connection_string: {connection_string}")
                dprint(f"🔍 SQLAnalysisTool: Extracted database_type: {database_type}")
        
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
        
        # NOTE: SQL validation will happen AFTER SQL generation, not on the user query
        
        try:
            # Import required modules
            import duckdb
            from sqlalchemy import create_engine, text
            from sqlalchemy.engine import Engine
            from langchain_community.utilities.sql_database import SQLDatabase
            from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            from langchain_community.agent_toolkits.sql.base import create_sql_agent
            
            # Use centralized LLM manager instead of creating new instances
            llm = llm_manager.get_llm(temperature=0.0)
            
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
                    dprint(f"🔍 SQLAnalysisTool: Creating view 'data' from S3 path: {s3_path}")
                    
                    try:
                        # Test if we can read from S3 first
                        test_query = f"SELECT COUNT(*) FROM read_parquet('{s3_path}') LIMIT 5"
                        dprint(f"🔍 SQLAnalysisTool: Testing S3 access with: {test_query}")
                        test_result = db.execute(test_query).fetchone()
                        dprint(f"🔍 SQLAnalysisTool: S3 test successful, count: {test_result}")
                        
                        # Now create the view
                        create_view_sql = f"CREATE VIEW data AS SELECT * FROM read_parquet('{s3_path}')"
                        dprint(f"🔍 SQLAnalysisTool: Creating view with: {create_view_sql}")
                        db.execute(create_view_sql)
                        dprint(f"🔍 SQLAnalysisTool: View 'data' created successfully")
                        
                    except Exception as view_error:
                        dprint(f"❌ SQLAnalysisTool: Failed to create view: {view_error}")
                        return f"Error: Failed to create data view from S3: {str(view_error)}"
                    
                    # First, inspect the schema to get actual column names
                    try:
                        dprint(f"🔍 SQLAnalysisTool: Inspecting view schema...")
                        schema_result = db.execute("DESCRIBE data").fetchdf()
                        dprint(f"🔍 SQLAnalysisTool: Schema inspection successful, columns: {len(schema_result)}")
                        
                        sample_result = db.execute("SELECT * FROM data LIMIT 3").fetchdf()
                        dprint(f"🔍 SQLAnalysisTool: Sample data retrieval successful, rows: {len(sample_result)}")
                        
                        # Create a detailed schema description
                        schema_info = f"Available columns: {', '.join(schema_result['column_name'].tolist())}\n"
                        schema_info += f"Column types: {dict(zip(schema_result['column_name'], schema_result['column_type']))}\n"
                        schema_info += f"Sample data:\n{sample_result.to_string()}"
                        
                        # Use LLM to generate SQL query with actual schema
                        # Detect database type and get appropriate hints
                        detected_db_type = sql_mapper.detect_database_type(connection_string)
                        db_hints = sql_mapper.get_database_specific_hints(detected_db_type)
                        
                        # Check cache first
                        query_hash = sql_query_cache.generate_query_hash(query, detected_db_type, connection_string)
                        cached_result = await sql_query_cache.get_cached_query(query_hash)
                        if cached_result:
                            dprint(f"🔍 SQLAnalysisTool: Using cached result for query")
                            return f"Query executed successfully (cached). Result: {cached_result['result']}"
                        
                        # Use LLM to generate SQL queries for all questions in JSON format using ChatPromptTemplate
                        system_template = """You are a SQL expert. Generate SQL queries to answer data analysis questions.

CRITICAL REQUIREMENTS:
- **CRITICAL: Use ONLY the exact column names from the provided schema - NO EXCEPTIONS**
- **CRITICAL: Do NOT invent, guess, or modify column names in any way**
- **CRITICAL: If the schema shows 'court', use 'court' - NOT 'court_code', NOT 'court_id', NOT anything else**
- Follow the database-specific hints for date functions
- **CRITICAL: Use the table name 'data' - this is the view we created from the S3 data**
- **DO NOT use any other table names like 'employees', 'users', etc.**
- **CRITICAL: Use SINGLE QUOTES (') for ALL string literals, including JSON_OBJECT keys**
- **Example: SELECT JSON_OBJECT('key1', 'value1', 'key2', 'value2') NOT SELECT JSON_OBJECT("key1", "value1")**
- **CRITICAL: Do NOT escape quotes with backslashes - use raw single quotes only**
- **Example: Use 'key' NOT \'key\' or \\'key\\'**
- **CRITICAL: For regression analysis, use REGR_SLOPE(y, x) instead of SLOPE(y, x)**
- **CRITICAL: For date differences, use DATE_DIFF('day', start_date, end_date) instead of JULIANDAY**
- **CRITICAL: RESPECT ALL LIMIT CLAUSES from user queries - if user asks for LIMIT 10, include LIMIT 10 in SQL**
- **CRITICAL: Before returning any SQL, validate that you are using ONLY the exact column names from the provided schema**
- **CRITICAL: ALWAYS use AS aliases for calculated columns to avoid syntax errors**
- **CRITICAL: Example: SELECT DATE_DIFF('day', start_date, end_date) AS delay_days, NOT just DATE_DIFF(...)**

Return ONLY the JSON object, no explanations, no markdown formatting."""

                        human_template = """Generate SQL queries to answer ALL questions in this request: {query}

**CRITICAL: The data is available in a view called 'data' - you MUST use 'data' as the table name**
                        
The data is available in a view called 'data' with the following ACTUAL schema:
{schema_info}

**SCHEMA VALIDATION: Before writing any SQL, verify that you are using EXACTLY these column names from the schema above. Do NOT use any other column names.**

DATABASE TYPE: {detected_db_type}
DATABASE-SPECIFIC HINTS:
{db_hints}

ANALYSIS TYPE GUIDANCE:
- For REGRESSION ANALYSIS or TREND ANALYSIS: You need individual data points, NOT aggregated by year
  - Use SELECT without GROUP BY to get individual cases
  - **EXAMPLE: SELECT date_of_registration, decision_date, year FROM data WHERE court = '33_10' AND year BETWEEN 2019 AND 2022**
  - **NOTE: Use 'court' (from schema) NOT 'court_code' or any other column name**
  - This provides multiple rows for regression analysis

- For COUNTING/AGGREGATION: Use GROUP BY with COUNT(), SUM(), AVG() as appropriate

- For RAW DATA: Use simple SELECT with WHERE conditions (only when specifically requested for individual records)

**CRITICAL: ALWAYS use AS aliases for calculated columns to avoid syntax errors**
**CRITICAL: Example: SELECT DATE_DIFF('day', start_date, end_date) AS delay_days, NOT just DATE_DIFF(...)**

**IMPORTANT: Return a JSON object with ALL SQL queries, one for each question.**

Return format:
{{
  "queries": [
    {{
      "question": "Question description",
      "sql": "SELECT ... FROM data WHERE ..."
    }},
    {{
      "question": "Question description", 
      "sql": "SELECT ... FROM data WHERE ..."
    }}
  ]
}}"""

                        # Create the prompt template
                        sql_prompt_template = ChatPromptTemplate.from_messages([
                            SystemMessagePromptTemplate.from_template(system_template),
                            HumanMessagePromptTemplate.from_template(human_template)
                        ])
                        
                        # Format the prompt with variables
                        formatted_sql_prompt = sql_prompt_template.format_messages(
                            query=query,
                            schema_info=schema_info,
                            detected_db_type=detected_db_type.upper(),
                            db_hints=db_hints
                        )
                        
                        # Convert to string format for the LLM
                        sql_prompt = ""
                        for message in formatted_sql_prompt:
                            if message.type == "system":
                                sql_prompt += f"SYSTEM: {message.content}\n\n"
                            elif message.type == "human":
                                sql_prompt += f"USER: {message.content}\n\n"
                        
                                                # Get SQL queries from LLM using centralized manager
                        sql_response = await llm.ainvoke(sql_prompt)
                        sql_response_content = sql_response.content
                        
                        dprint(f"🔍 SQLAnalysisTool: Raw LLM response: {sql_response_content[:200]}...")
                        
                        # Parse the JSON response to get individual SQL queries
                        try:
                            # For JSON responses, we need to clean markdown but preserve JSON structure
                            cleaned_response = sql_response_content.strip()
                            
                            # Remove markdown code blocks but preserve JSON content
                            if cleaned_response.startswith('```json'):
                                cleaned_response = cleaned_response[7:]
                            elif cleaned_response.startswith('```'):
                                cleaned_response = cleaned_response[3:]
                            if cleaned_response.endswith('```'):
                                cleaned_response = cleaned_response[:-3]
                            cleaned_response = cleaned_response.strip()
                            
                            dprint(f"🔍 SQLAnalysisTool: Cleaned JSON response: {cleaned_response[:200]}...")
                            
                            # Parse JSON to get queries array
                            queries_data = json.loads(cleaned_response)
                            
                            if 'queries' not in queries_data or not isinstance(queries_data['queries'], list):
                                raise ValueError("Invalid JSON format - missing 'queries' array")
                            
                            queries = queries_data['queries']
                            dprint(f"🔍 SQLAnalysisTool: Parsed {len(queries)} SQL queries from LLM response")
                            
                        except Exception as parse_error:
                            dprint(f"🔍 SQLAnalysisTool: Failed to parse JSON response: {parse_error}")
                            dprint(f"🔍 SQLAnalysisTool: Raw response: {sql_response_content}")
                            return f"Error: Failed to parse SQL queries from LLM response: {str(parse_error)}"
                        
                        # Execute each query individually and collect results
                        all_results = []
                        failed_queries = []
                        
                        # First pass: Execute all queries
                        for i, query_item in enumerate(queries):
                            try:
                                question = query_item.get('question', f'Question {i+1}')
                                sql_query = query_item.get('sql', '')
                                
                                if not sql_query:
                                    dprint(f"⚠️ SQLAnalysisTool: Query {i+1} has no SQL")
                                    continue
                                
                                dprint(f"🔍 SQLAnalysisTool: Executing query {i+1}: {question}")
                                dprint(f"🔍 SQLAnalysisTool: SQL: {sql_query[:200]}...")
                                
                                # CRITICAL FIX: Validate the generated SQL query for security and syntax
                                try:
                                    # Security validation
                                    if self._contains_dangerous_operations(sql_query):
                                        dprint(f"❌ SQLAnalysisTool: Query {i+1} contains dangerous operations")
                                        all_results.append({
                                            'question': question,
                                            'sql': adapted_sql,
                                            'error': 'SQL contains dangerous operations (DROP, DELETE, INSERT, UPDATE, etc.). Only SELECT queries are allowed.',
                                            'status': 'error'
                                        })
                                        failed_queries.append({'index': i, 'item': query_item, 'reason': 'dangerous_operations'})
                                        continue
                                    
                                    # Basic syntax validation
                                    if not self._validate_basic_syntax(sql_query):
                                        dprint(f"❌ SQLAnalysisTool: Query {i+1} has syntax issues")
                                        all_results.append({
                                            'question': question,
                                            'sql': adapted_sql,
                                            'error': 'SQL has basic syntax issues. Please check parentheses, quotes, and SQL structure.',
                                            'status': 'error'
                                        })
                                        failed_queries.append({'index': i, 'item': query_item, 'reason': 'syntax_issues'})
                                        continue
                                    
                                    dprint(f"🔍 SQLAnalysisTool: Query {i+1} passed security and syntax validation")
                                    
                                except Exception as validation_error:
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} validation failed: {validation_error}")
                                    all_results.append({
                                        'question': question,
                                        'sql': adapted_sql,
                                        'error': f'SQL validation failed: {str(validation_error)}',
                                        'status': 'error'
                                    })
                                    failed_queries.append({'index': i, 'item': query_item, 'reason': 'validation_failed'})
                                    continue
                                
                                # Adapt SQL for the specific database type
                                adapted_sql = self._adapt_sql_for_database(sql_query, detected_db_type)
                                
                                dprint(f"🔍 SQLAnalysisTool: Query {i+1} adapted SQL: {adapted_sql[:200]}...")
                                
                                # Execute with timeout protection - use the SAME DuckDB connection that has the view
                                try:
                                    async def execute_query():
                                        # Use the SAME DuckDB connection that has the 'data' view
                                        return db.execute(adapted_sql).fetchdf()
                                    
                                    result = await sql_timeout_manager.execute_with_timeout(
                                        execute_query, 
                                        timeout_seconds=120,  # Increased from 30s to 120s for complex queries
                                        query_id=f"duckdb_s3_{query_hash[:8]}_{i}"
                                    )
                                    
                                    # Process the result
                                    if hasattr(result, 'to_dict'):
                                        result_dict = result.to_dict('records')
                                        # Limit to first 100 rows to prevent JSON parsing issues
                                        if len(result_dict) > 100:
                                            result_dict = result_dict[:100]
                                            result_summary = f"Result: {len(result_dict)} rows (showing first 100). Data: {result_dict}"
                                        else:
                                            result_summary = f"Result: {len(result_dict)} rows. Data: {result_dict}"
                                    else:
                                        result_summary = f"Result: {result}"
                                    
                                    all_results.append({
                                        'question': question,
                                        'sql': adapted_sql,
                                        'result': result_summary,
                                        'status': 'success'
                                    })
                                    
                                    dprint(f"✅ SQLAnalysisTool: Query {i+1} executed successfully")
                                    
                                except TimeoutError as e:
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} timed out: {e}")
                                    all_results.append({
                                        'question': question,
                                        'sql': adapted_sql,
                                        'error': f'Query execution timed out after 120 seconds',
                                        'status': 'timeout'
                                    })
                                    failed_queries.append({'index': i, 'item': query_item, 'reason': 'timeout'})
                                    
                                except Exception as e:
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} execution failed: {e}")
                                    all_results.append({
                                        'question': question,
                                        'sql': adapted_sql,
                                        'error': f'Query execution failed: {str(e)}',
                                        'status': 'error'
                                    })
                                    failed_queries.append({'index': i, 'item': query_item, 'reason': 'execution_failed'})
                                    
                            except Exception as query_error:
                                dprint(f"❌ SQLAnalysisTool: Error processing query {i+1}: {query_error}")
                                all_results.append({
                                    'question': f'Question {i+1}',
                                    'sql': sql_query,
                                    'error': f'Error processing query: {str(query_error)}',
                                    'status': 'error'
                                })
                                failed_queries.append({'index': i, 'item': query_item, 'reason': 'processing_error'})
                        
                        # Retry failed queries if any
                        if failed_queries:
                            dprint(f"🔄 SQLAnalysisTool: {len(failed_queries)} queries failed, attempting retry...")
                            
                            for failed_item in failed_queries:
                                i = failed_item['index']
                                query_item = failed_item['item']
                                reason = failed_item['reason']
                                question = query_item.get('question', f'Question {i+1}')
                                sql_query = query_item.get('sql', '')
                                
                                dprint(f"🔄 SQLAnalysisTool: Retrying query {i+1} (failed due to: {reason}): {question}")
                                
                                try:
                                    # Adapt SQL for the specific database type
                                    adapted_sql = self._adapt_sql_for_database(sql_query, detected_db_type)
                                    
                                    # Execute with timeout protection
                                    async def execute_query():
                                        return db.execute(adapted_sql).fetchdf()
                                    
                                    result = await sql_timeout_manager.execute_with_timeout(
                                        execute_query, 
                                        timeout_seconds=120,
                                        query_id=f"duckdb_s3_retry_{query_hash[:8]}_{i}"
                                    )
                                    
                                    # Process the result
                                    if hasattr(result, 'to_dict'):
                                        result_dict = result.to_dict('records')
                                        if len(result_dict) > 100:
                                            result_dict = result_dict[:100]
                                            result_summary = f"Result: {len(result_dict)} rows (showing first 100). Data: {result_dict}"
                                        else:
                                            result_summary = f"Result: {len(result_dict)} rows. Data: {result_dict}"
                                    else:
                                        result_summary = f"Result: {result}"
                                    
                                    # Update the failed result with success
                                    for j, existing_result in enumerate(all_results):
                                        if existing_result.get('question') == question and existing_result.get('status') == 'error':
                                            all_results[j] = {
                                                'question': question,
                                                'sql': adapted_sql,
                                                'result': result_summary,
                                                'status': 'success_after_retry'
                                            }
                                            break
                                    
                                    dprint(f"✅ SQLAnalysisTool: Query {i+1} retry successful")
                                    
                                except Exception as retry_error:
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} retry also failed: {retry_error}")
                                    # Keep the original error result
                        
                        # Cache the successful results
                        await sql_query_cache.cache_query(query_hash, str(all_results), all_results, detected_db_type)
                        
                        # Store the results in data cache for potential plotting tasks
                        if 'data_cache' in kwargs:
                            await kwargs['data_cache'].store("latest_sql_result", all_results)
                        
                        # Return combined results with retry summary
                        successful_queries = [r for r in all_results if r.get('status') == 'success']
                        retry_successful = [r for r in all_results if r.get('status') == 'success_after_retry']
                        failed_queries_final = [r for r in all_results if r.get('status') == 'error']
                        
                        summary = f"Execution Summary: {len(successful_queries)} succeeded initially, {len(retry_successful)} succeeded after retry, {len(failed_queries_final)} failed"
                        
                        return f"{summary}\n\nAll SQL queries executed. Results: {json.dumps(all_results, indent=2)}"
                    except Exception as sql_error:
                        dprint(f"🔍 SQLAnalysisTool: SQL execution error: {str(sql_error)}")
                        dprint(f"🔍 SQLAnalysisTool: Error type: {type(sql_error).__name__}")
                        return f"SQL execution error: {str(sql_error)}. Please check the query syntax and database connection."
                    
                else:
                    # For local DuckDB files
                    db = duckdb.connect(connection_string or ':memory:')
                    
                    # Use LLM to generate SQL queries for all questions in JSON format
                    sql_prompt = f"""
                    You are a SQL expert. Generate SQL queries to answer ALL questions in this request: {query}
                    
                    ANALYSIS TYPE GUIDANCE:
                    - For REGRESSION ANALYSIS or TREND ANALYSIS: You need individual data points, NOT aggregated by year
                      - Use SELECT without GROUP BY to get individual cases
                      - Example: SELECT date_of_registration, decision_date, year FROM data WHERE court = '33_10' AND year BETWEEN 2019 AND 2022
                      - This provides multiple rows for regression analysis
                    
                    - For COUNTING/AGGREGATION: Use GROUP BY with COUNT(), SUM(), AVG() as appropriate
                    
                    - For RAW DATA: Use simple SELECT with WHERE conditions (only when specifically requested for individual records)
                    
                    CRITICAL REQUIREMENTS:
                    - **CRITICAL: Use the table name 'data' - this is the view we created from the S3 data**
                    - **CRITICAL: RESPECT ALL LIMIT CLAUSES from user queries - if user asks for LIMIT 10, include LIMIT 10 in SQL**
                    - **CRITICAL: ALWAYS use AS aliases for calculated columns to avoid syntax errors**
                    - **CRITICAL: Example: SELECT DATE_DIFF('day', start_date, end_date) AS delay_days, NOT just DATE_DIFF(...)**
                    
                    **IMPORTANT: Return a JSON object with ALL SQL queries, one for each question.**
                    
                    Return format:
                    {{
                      "queries": [
                        {{
                          "question": "Question description",
                          "sql": "SELECT ... FROM data WHERE ..."
                        }},
                        {{
                          "question": "Question description", 
                          "sql": "SELECT ... FROM data WHERE ..."
                        }}
                      ]
                    }}
                    
                    Return ONLY the JSON object, no explanations, no markdown formatting.
                    """
                    
                    # Use centralized SQL generation and validation
                    sql_response = await llm.ainvoke(sql_prompt)
                    sql_response_content = sql_response.content
                    
                    dprint(f"🔍 SQLAnalysisTool: Raw LLM response (local): {sql_response_content[:200]}...")
                    
                    # Parse the JSON response to get individual SQL queries
                    try:
                        # For JSON responses, we need to clean markdown but preserve JSON structure
                        cleaned_response = sql_response_content.strip()
                        
                        # Remove markdown code blocks but preserve JSON content
                        if cleaned_response.startswith('```json'):
                            cleaned_response = cleaned_response[7:]
                        elif cleaned_response.startswith('```'):
                            cleaned_response = cleaned_response[3:]
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response[:-3]
                        cleaned_response = cleaned_response.strip()
                        
                        dprint(f"🔍 SQLAnalysisTool: Cleaned JSON response (local): {cleaned_response[:200]}...")
                        
                        # Parse JSON to get queries array
                        queries_data = json.loads(cleaned_response)
                        
                        if 'queries' not in queries_data or not isinstance(queries_data['queries'], list):
                            raise ValueError("Invalid JSON format - missing 'queries' array")
                        
                        queries = queries_data['queries']
                        dprint(f"🔍 SQLAnalysisTool: Parsed {len(queries)} SQL queries from LLM response (local)")
                        
                    except Exception as parse_error:
                        dprint(f"🔍 SQLAnalysisTool: Failed to parse JSON response (local): {parse_error}")
                        dprint(f"🔍 SQLAnalysisTool: Raw response: {sql_response_content}")
                        return f"Error: Failed to parse SQL queries from LLM response: {str(parse_error)}"
                    
                    # Execute each query individually and collect results
                    all_results = []
                    for i, query_item in enumerate(queries):
                        try:
                            question = query_item.get('question', f'Question {i+1}')
                            sql_query = query_item.get('sql', '')
                            
                            if not sql_query:
                                dprint(f"⚠️ SQLAnalysisTool: Query {i+1} has no SQL (local)")
                                continue
                            
                            dprint(f"🔍 SQLAnalysisTool: Executing query {i+1} (local): {question}")
                            dprint(f"🔍 SQLAnalysisTool: SQL: {sql_query[:200]}...")
                            
                            # CRITICAL FIX: Validate the generated SQL query for security and syntax
                            try:
                                # Security validation
                                if self._contains_dangerous_operations(sql_query):
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} contains dangerous operations (local)")
                                    all_results.append({
                                        'question': question,
                                        'error': 'SQL contains dangerous operations (DROP, DELETE, INSERT, UPDATE, etc.). Only SELECT queries are allowed.'
                                    })
                                    continue
                                
                                # Basic syntax validation
                                if not self._validate_basic_syntax(sql_query):
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} has syntax issues (local)")
                                    all_results.append({
                                        'question': question,
                                        'error': 'SQL has basic syntax issues. Please check parentheses, quotes, and SQL structure.'
                                    })
                                    continue
                                
                                dprint(f"🔍 SQLAnalysisTool: Query {i+1} passed security and syntax validation (local)")
                                
                            except Exception as validation_error:
                                dprint(f"❌ SQLAnalysisTool: Query {i+1} validation failed (local): {validation_error}")
                                all_results.append({
                                    'question': question,
                                    'error': f'SQL validation failed: {str(validation_error)}'
                                })
                                continue
                            
                            # Adapt SQL for the specific database type
                            adapted_sql = self._adapt_sql_for_database(sql_query, database_type)
                            
                            dprint(f"🔍 SQLAnalysisTool: Query {i+1} adapted SQL (local): {adapted_sql[:200]}...")
                            
                            # Execute with timeout protection
                            try:
                                async def execute_query():
                                    return db.execute(adapted_sql).fetchdf()
                                
                                result = await sql_timeout_manager.execute_with_timeout(
                                    execute_query, 
                                    timeout_seconds=120,  # Increased from 30s to 120s for complex queries
                                    query_id=f"duckdb_local_{hash(query) % 10000}_{i}"
                                )
                                
                                # Process the result
                                if hasattr(result, 'to_dict'):
                                    result_dict = result.to_dict('records')
                                    # Limit to first 100 rows to prevent JSON parsing issues
                                    if len(result_dict) > 100:
                                        result_dict = result_dict[:100]
                                        result_summary = f"Result: {len(result_dict)} rows (showing first 100). Data: {result_dict}"
                                    else:
                                        result_summary = f"Result: {len(result_dict)} rows. Data: {result_dict}"
                                else:
                                    result_summary = f"Result: {result}"
                                
                                all_results.append({
                                    'question': question,
                                    'sql': adapted_sql,
                                    'result': result_summary,
                                    'status': 'success'
                                })
                                
                                dprint(f"✅ SQLAnalysisTool: Query {i+1} executed successfully (local)")
                                
                            except TimeoutError as e:
                                dprint(f"❌ SQLAnalysisTool: Query {i+1} timed out (local): {e}")
                                all_results.append({
                                    'question': question,
                                    'sql': adapted_sql,
                                    'error': f'Query execution timed out after 120 seconds',
                                    'status': 'timeout'
                                })
                                
                            except Exception as e:
                                dprint(f"❌ SQLAnalysisTool: Query {i+1} execution failed (local): {e}")
                                all_results.append({
                                    'question': question,
                                    'sql': adapted_sql,
                                    'error': f'Query execution failed: {str(e)}',
                                    'status': 'error'
                                })
                                
                        except Exception as query_error:
                            dprint(f"❌ SQLAnalysisTool: Error processing query {i+1} (local): {query_error}")
                            all_results.append({
                                'question': f'Question {i+1}',
                                'error': f'Error processing query: {str(query_error)}',
                                'status': 'error'
                            })
                    
                    # Store the results in data cache for potential plotting tasks
                    if 'data_cache' in kwargs:
                        await kwargs['data_cache'].store("latest_sql_result", all_results)
                    
                    # Return combined results
                    return f"All SQL queries executed. Results: {json.dumps(all_results, indent=2)}"
                    
            else:
                # For other SQL databases, use the connection string directly
                if not connection_string:
                    return "Error: Connection string required for non-DuckDB databases"
                
                try:
                    # Use LLM to generate SQL queries for all questions in JSON format using ChatPromptTemplate
                    system_template = """You are a SQL expert. Generate SQL queries to answer data analysis questions.

CRITICAL REQUIREMENTS:
- **CRITICAL: Use the table name 'data' - this is the view we created from the S3 data**
- **CRITICAL: RESPECT ALL LIMIT CLAUSES from user queries - if user asks for LIMIT 10, include LIMIT 10 in SQL**
- **CRITICAL: ALWAYS use AS aliases for calculated columns to avoid syntax errors**
- **CRITICAL: Example: SELECT DATE_DIFF('day', start_date, end_date) AS delay_days, NOT just DATE_DIFF(...)**

Return ONLY the JSON object, no explanations, no markdown formatting."""

                    human_template = """Generate SQL queries to answer ALL questions in this request: {query}

ANALYSIS TYPE GUIDANCE:
- For REGRESSION ANALYSIS or TREND ANALYSIS: You need individual data points, NOT aggregated by year
  - Use SELECT without GROUP BY to get individual cases
  - Example: SELECT date_of_registration, decision_date, year FROM data WHERE court = '33_10' AND year BETWEEN 2019 AND 2022
  - This provides multiple rows for regression analysis

- For COUNTING/AGGREGATION: Use GROUP BY with COUNT(), SUM(), AVG() as appropriate

- For RAW DATA: Use simple SELECT with WHERE conditions (only when specifically requested for individual records)

**CRITICAL: ALWAYS use AS aliases for calculated columns to avoid syntax errors**
**CRITICAL: Example: SELECT DATE_DIFF('day', start_date, end_date) AS delay_days, NOT just DATE_DIFF(...)**

**IMPORTANT: Return a JSON object with ALL SQL queries, one for each question.**

Return format:
{{
  "queries": [
    {{
      "question": "Question description",
      "sql": "SELECT ... FROM data WHERE ..."
    }},
    {{
      "question": "Question description", 
      "sql": "SELECT ... FROM data WHERE ..."
    }}
  ]
}}"""

                    # Create the prompt template
                    sql_prompt_template = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_template),
                        HumanMessagePromptTemplate.from_template(human_template)
                    ])
                    
                    # Format the prompt with variables
                    formatted_sql_prompt = sql_prompt_template.format_messages(query=query)
                    
                    # Convert to string format for the LLM
                    sql_prompt = ""
                    for message in formatted_sql_prompt:
                        if message.type == "system":
                            sql_prompt += f"SYSTEM: {message.content}\n\n"
                        elif message.type == "human":
                            sql_prompt += f"USER: {message.content}\n\n"
                    
                    # Use centralized SQL generation and validation
                    sql_response = await llm.ainvoke(sql_prompt)
                    sql_response_content = sql_response.content
                    
                    dprint(f"🔍 SQLAnalysisTool: Raw LLM response (other): {sql_response_content[:200]}...")
                    
                    # Parse the JSON response to get individual SQL queries
                    try:
                        # For JSON responses, we need to clean markdown but preserve JSON structure
                        cleaned_response = sql_response_content.strip()
                        
                        # Remove markdown code blocks but preserve JSON content
                        if cleaned_response.startswith('```json'):
                            cleaned_response = cleaned_response[7:]
                        elif cleaned_response.startswith('```'):
                            cleaned_response = cleaned_response[3:]
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response[:-3]
                        cleaned_response = cleaned_response.strip()
                        
                        dprint(f"🔍 SQLAnalysisTool: Cleaned JSON response (other): {cleaned_response[:200]}...")
                        
                        # Parse JSON to get queries array
                        queries_data = json.loads(cleaned_response)
                        
                        if 'queries' not in queries_data or not isinstance(queries_data['queries'], list):
                            raise ValueError("Invalid JSON format - missing 'queries' array")
                        
                        queries = queries_data['queries']
                        dprint(f"🔍 SQLAnalysisTool: Parsed {len(queries)} SQL queries from LLM response (other)")
                        
                    except Exception as parse_error:
                        dprint(f"🔍 SQLAnalysisTool: Failed to parse JSON response (other): {parse_error}")
                        dprint(f"🔍 SQLAnalysisTool: Raw response: {sql_response_content}")
                        return f"Error: Failed to parse SQL queries from LLM response: {str(parse_error)}"
                    
                    # Execute each query individually and collect results
                    all_results = []
                    for i, query_item in enumerate(queries):
                        try:
                            question = query_item.get('question', f'Question {i+1}')
                            sql_query = query_item.get('sql', '')
                            
                            if not sql_query:
                                dprint(f"⚠️ SQLAnalysisTool: Query {i+1} has no SQL (other)")
                                continue
                            
                            dprint(f"🔍 SQLAnalysisTool: Executing query {i+1} (other): {question}")
                            dprint(f"🔍 SQLAnalysisTool: SQL: {sql_query[:200]}...")
                            
                            # CRITICAL FIX: Validate the generated SQL query for security and syntax
                            try:
                                # Security validation
                                if self._contains_dangerous_operations(sql_query):
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} contains dangerous operations (other)")
                                    all_results.append({
                                        'question': question,
                                        'error': 'SQL contains dangerous operations (DROP, DELETE, INSERT, UPDATE, etc.). Only SELECT queries are allowed.'
                                    })
                                    continue
                                
                                # Basic syntax validation
                                if not self._validate_basic_syntax(sql_query):
                                    dprint(f"❌ SQLAnalysisTool: Query {i+1} has syntax issues (other)")
                                    all_results.append({
                                        'question': question,
                                        'error': 'SQL has basic syntax issues. Please check parentheses, quotes, and SQL structure.'
                                    })
                                    continue
                                
                                dprint(f"🔍 SQLAnalysisTool: Query {i+1} passed security and syntax validation (other)")
                                
                            except Exception as validation_error:
                                dprint(f"❌ SQLAnalysisTool: Query {i+1} validation failed (other): {validation_error}")
                                all_results.append({
                                    'question': question,
                                    'error': f'SQL validation failed: {str(validation_error)}'
                                })
                                continue
                            
                            # Adapt SQL for the specific database type
                            adapted_sql = self._adapt_sql_for_database(sql_query, database_type)
                            
                            dprint(f"🔍 SQLAnalysisTool: Query {i+1} adapted SQL (other): {adapted_sql[:200]}...")
                            
                            # Execute with timeout protection and connection pooling
                            try:
                                # Get connection from pool
                                db_connection = await sql_connection_pool.get_connection(connection_string, database_type)
                                
                                async def execute_query():
                                    with db_connection.connect() as connection:
                                        result = connection.execute(text(adapted_sql))
                                        rows = result.fetchall()
                                        columns = result.keys()
                                        return [dict(zip(columns, row)) for row in rows]
                                
                                result_data = await sql_timeout_manager.execute_with_timeout(
                                    execute_query, 
                                    timeout_seconds=120,  # Increased from 30s to 120s for complex queries
                                    query_id=f"sql_{database_type}_{hash(query) % 10000}_{i}"
                                )
                                
                                # Return connection to pool
                                await sql_connection_pool.return_connection(db_connection, connection_string, database_type)
                                
                                # Process the result
                                if isinstance(result_data, list):
                                    result_summary = f"Result: {len(result_data)} rows. Data: {result_data}"
                                else:
                                    result_summary = f"Result: {result_data}"
                                
                                all_results.append({
                                    'question': question,
                                    'sql': adapted_sql,
                                    'result': result_summary,
                                    'status': 'success'
                                })
                                
                                dprint(f"✅ SQLAnalysisTool: Query {i+1} executed successfully (other)")
                                
                            except TimeoutError as e:
                                dprint(f"❌ SQLAnalysisTool: Query {i+1} timed out (other): {e}")
                                all_results.append({
                                    'question': question,
                                    'sql': adapted_sql,
                                    'error': f'Query execution timed out after 120 seconds',
                                    'status': 'timeout'
                                })
                                
                            except Exception as e:
                                dprint(f"❌ SQLAnalysisTool: Query {i+1} execution failed (other): {e}")
                                all_results.append({
                                    'question': question,
                                    'sql': adapted_sql,
                                    'error': f'Query execution failed: {str(e)}',
                                    'status': 'error'
                                })
                                
                        except Exception as query_error:
                            dprint(f"❌ SQLAnalysisTool: Error processing query {i+1} (other): {query_error}")
                            all_results.append({
                                'question': f'Question {i+1}',
                                'error': f'Error processing query: {str(query_error)}',
                                'status': 'error'
                            })
                    
                    # Store the results in data cache for potential plotting tasks
                    if 'data_cache' in kwargs:
                        await kwargs['data_cache'].store("latest_sql_result", all_results)
                    
                    # Return combined results
                    return f"All SQL queries executed. Results: {json.dumps(all_results, indent=2)}"
                    
                except Exception as e:
                    return f"Error in SQL generation: {str(e)}"
                
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
                agent = create_react_agent(llm=llm_manager.get_llm(temperature=0.0), tools=tools, verbose=True)
                
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
                agent = create_react_agent(llm=llm_manager.get_llm(temperature=0.0), tools=tools, verbose=True)
                
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

async def create_planner_executor_workflow(user_query: str, file_path: Optional[str] = None, custom_format: str = None):
    """
    Creates and executes a workflow using the new Planner-Executor architecture.
    This handles complex, multi-part queries with intelligent task planning and parallel execution.
    """
    try:
        # Use centralized LLM manager instead of creating new instances
        try:
            llm = llm_manager.get_llm(temperature=0.0)
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {str(e)}")
        
        # Initialize the data cache for sharing data between tasks
        data_cache = DataCache()
        
        # Create the Planner Agent to generate execution plan
        planner = PlannerAgent(llm)
        
        dprint("🧠 Creating execution plan...")
        execution_plan = await planner.create_execution_plan(user_query, file_path)
        
        # Add custom_format to the execution plan context if it exists
        if custom_format:
            execution_plan.context['custom_format'] = custom_format

        dprint(f"📋 Execution plan created with {len(execution_plan.stages)} stages")
        dprint(f"📋 Execution plan context: {execution_plan.context}")
        dprint(f"📋 User query that was used for planning: {user_query[:200]}...")
        for stage in execution_plan.stages:
            dprint(f"  Stage {stage.stage_id}: {stage.description} ({len(stage.tasks)} tasks)")
            for task in stage.tasks:
                dprint(f"    - {task.task_id}: {task.description}")
                dprint(f"      Tool: {task.tool}")
                dprint(f"      Context: {task.context}")
        
        # Create the Executor Crew to execute the plan
        # Initialize with all available tools
        tools = {
            "ScrapingTool": ScrapingTool(),
            "DataframeAnalysisTool": DataframeAnalysisTool(),
            "PDFAnalysisTool": PDFAnalysisTool(),
            "SQLAnalysisTool": SQLAnalysisTool(),  # No need to pass llm - uses centralized manager
            "NoSQLAnalysisTool": NoSQLAnalysisTool()
        }
        executor = ExecutorCrew(data_cache, tools)
        
        dprint("🚀 Executing plan...")
        dprint(f"🚀 Passing execution plan to ExecutorCrew with user_query: {user_query[:200]}...")
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
            # Multiple results: combine SQL and DataFrame outputs in order
            combined_parts: list[str] = []
            # Prefer any SQLAnalysisTool outputs first
            for task_id, result in final_results.items():
                if isinstance(result, str) and "SQL" in task_id.lower():
                    combined_parts.append(result)
            # Then append DataframeAnalysisTool outputs
            for task_id, result in final_results.items():
                if isinstance(result, str) and ("dataframe" in task_id.lower() or "plot" in result.lower() or "image_path" in result.lower() or "base64" in result.lower()):
                    combined_parts.append(result)
            # Fallback to include any remaining string results
            if not combined_parts:
                for _task_id, res in final_results.items():
                    if isinstance(res, str):
                        combined_parts.append(res)
            combined_answer = "\n\n".join(combined_parts) if combined_parts else str(final_results)
            return combined_answer, ["Planner-Executor workflow completed successfully"]
            
    except Exception as e:
        dprint(f"❌ Planner-Executor workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Planner-Executor workflow failed: {str(e)}"}, []

@app.get("/")
async def root():
    return {"message": "Data Analyst AI Agent API", "version": "1.0.0"}

@app.post("/analyze")
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

        # Detect custom answer format from query text (if any)
        try:
            custom_format = await ResponseFormatter.extract_answer_format_from_file(final_query)
            if custom_format:
                dprint(f"📋 Detected custom answer format from query: {custom_format}")
        except Exception as _:
            custom_format = None

        # Execute the workflow using the new Planner-Executor architecture
        result, reasoning_steps = await create_planner_executor_workflow(
            final_query,
            file_path=temp_file_path,
            custom_format=custom_format
        )
        
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
    
        
        # Use ResponseFormatter to ensure consistent JSON array format
        if custom_format:
            formatted_answer = await ResponseFormatter.format_final_answer(final_answer, final_query, custom_format)
        else:
            formatted_answer = await ResponseFormatter.format_final_answer(final_answer, final_query)
        
        # Return only the formatted answer as requested
        return formatted_answer
        
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

@app.post("/analyze_file")
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
        
        # Check for custom answer format specification in the file
        custom_format = await ResponseFormatter.extract_answer_format_from_file(file_content)
        if custom_format:
            dprint(f"📋 Detected custom answer format: {custom_format}")
        
        # Execute the Planner-Executor workflow with the file content
        result, reasoning_steps = await create_planner_executor_workflow(
            file_content, 
            file_path=temp_file_path, 
            custom_format=custom_format
        )
        
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
        
        # Use ResponseFormatter to ensure consistent JSON array format
        # Pass the custom format if detected
        
        # Simple bypass: if it looks like a JSON array, return it directly
        if isinstance(final_answer, str) and final_answer.strip().startswith('[') and final_answer.strip().endswith(']'):
            return final_answer
        
        # Check if it's a JSON object with an answer field that contains a JSON array
        if isinstance(final_answer, str):
            try:
                parsed = json.loads(final_answer)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer_content = parsed["answer"]
                    # Check if the answer field contains a JSON array
                    if isinstance(answer_content, str) and answer_content.strip().startswith('[') and answer_content.strip().endswith(']'):
                        return answer_content
            except json.JSONDecodeError:
                pass
        
        if custom_format:
            formatted_answer = await ResponseFormatter.format_final_answer(final_answer, file_content, custom_format)
        else:
            formatted_answer = await ResponseFormatter.format_final_answer(final_answer, file_content)
        
        # Return only the formatted answer as requested
        return formatted_answer
        
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
                        dprint(f"⚠️ Error deleting {file_path}: {e}")
            dprint(f"🧹 Cleaned up old table files from temp_files")
        else:
            # Create the directory if it doesn't exist
            os.makedirs("temp_files", exist_ok=True)
            dprint(f"📁 Created temp_files directory")
    except Exception as e:
        dprint(f"❌ Error during cleanup: {str(e)}")

class SQLFunctionMapper:
    """Comprehensive SQL function mapping for different database types"""
    
    def __init__(self):
        # Define function mappings for different database types
        self.function_maps = {
            'duckdb': {
                'date_functions': {
                    'JULIANDAY': 'DATE_DIFF',
                    'STRFTIME': 'STRPTIME',
                    'STRPTIME': 'STRPTIME',  # Keep as is for DuckDB
                },
                'date_patterns': [
                    # JULIANDAY(date1) - JULIANDAY(date2) → DATE_DIFF('day', date2, date1)
                    (r'JULIANDAY\s*\(([^)]+)\)\s*-\s*JULIANDAY\s*\(([^)]+)\)', 'DATE_DIFF(\'day\', \\2, \\1)'),
                    # JULIANDAY(date) → DATE_DIFF('day', '1970-01-01', date)
                    (r'JULIANDAY\s*\(([^)]+)\)', 'DATE_DIFF(\'day\', \'1970-01-01\', \\1)'),
                    # STRFTIME → STRPTIME for DuckDB
                    (r'STRFTIME\s*\(([^)]+)\)', r'STRPTIME(\1)'),
                    # Statistical functions → DuckDB-compatible alternatives
                    (r'(?<!REGR_)SLOPE\s*\(([^)]+)\)', 'REGR_SLOPE(\\1)'),
                    (r'(?<!REGR_)SLOPE\s*\(([^,]+),\s*([^)]+)\)', 'REGR_SLOPE(\\1, \\2)'),
                    (r'(?<!CORR)CORREL\s*\(([^)]+)\)', 'CORR(\\1)'),
                    (r'(?<!CORR)CORREL\s*\(([^,]+),\s*([^)]+)\)', 'CORR(\\1, \\2)'),
                    (r'(?<!REGR_)RSQUARED\s*\(([^)]+)\)', 'REGR_R2(\\1)'),
                    (r'(?<!REGR_)RSQUARED\s*\(([^,]+),\s*([^)]+)\)', 'REGR_R2(\\1, \\2)'),
                ],
                'constraints': {
                    'strptime_format': '%d-%m-%Y',  # DuckDB requires literal format strings
                }
            },
            'postgresql': {
                'date_functions': {
                    'JULIANDAY': 'EXTRACT(EPOCH FROM date_column) / 86400',
                    'STRFTIME': 'TO_CHAR',
                    'STRPTIME': 'TO_DATE',
                },
                'date_patterns': [
                    (r'JULIANDAY\s*\(([^)]+)\)', r'EXTRACT(EPOCH FROM \1) / 86400'),
                    (r'STRFTIME\s*\(([^,]+),\s*([^)]+)\)', r'TO_CHAR(\1, \2)'),
                    (r'STRPTIME\s*\(([^,]+),\s*([^)]+)\)', r'TO_DATE(\1, \2)'),
                    # SLOPE function → PostgreSQL-compatible REGR_SLOPE
                    (r'(?<!REGR_)SLOPE\s*\(([^)]+)\)', 'REGR_SLOPE(\\1)'),
                    (r'(?<!REGR_)SLOPE\s*\(([^,]+),\s*([^)]+)\)', 'REGR_SLOPE(\\1, \\2)'),
                ]
            },
            'mysql': {
                'date_functions': {
                    'JULIANDAY': 'TO_DAYS',
                    'STRFTIME': 'DATE_FORMAT',
                    'STRPTIME': 'STR_TO_DATE',
                },
                'date_patterns': [
                    (r'JULIANDAY\s*\(([^)]+)\)', r'TO_DAYS(\1)'),
                    (r'STRFTIME\s*\(([^,]+),\s*([^)]+)\)', r'DATE_FORMAT(\1, \2)'),
                    (r'STRPTIME\s*\(([^,]+),\s*([^)]+)\)', r'STR_TO_DATE(\1, \2)'),
                    # SLOPE function → MySQL-compatible (manual calculation needed)
                    (r'(?<!REGR_)SLOPE\s*\(([^)]+)\)', 'REGR_SLOPE(\\1)'),
                    (r'(?<!REGR_)SLOPE\s*\(([^,]+),\s*([^)]+)\)', 'REGR_SLOPE(\\1, \\2)'),
                ]
            },
            'sqlite': {
                'date_functions': {
                    'JULIANDAY': 'JULIANDAY',  # Keep as is for SQLite
                    'STRFTIME': 'STRFTIME',    # Keep as is for SQLite
                    'STRPTIME': 'STRPTIME',    # Keep as is for SQLite
                },
                'date_patterns': []  # No transformations needed for SQLite
            }
        }
    
    def detect_database_type(self, connection_string: str) -> str:
        """Detect database type from connection string or connection object"""
        if not connection_string:
            return 'duckdb'  # Default to DuckDB
        
        conn_str = str(connection_string).lower()
        
        if 'postgresql' in conn_str or 'postgres' in conn_str:
            return 'postgresql'
        elif 'mysql' in conn_str:
            return 'mysql'
        elif 'sqlite' in conn_str:
            return 'sqlite'
        elif 'duckdb' in conn_str or 's3://' in conn_str:
            return 'duckdb'
        else:
            return 'duckdb'  # Default fallback
    
    def adapt_sql_for_database(self, sql_query: str, database_type: str) -> str:
        """Adapt SQL query for specific database type"""
        if not database_type or database_type not in self.function_maps:
            dprint(f"⚠️ Unknown database type: {database_type}, using DuckDB defaults")
            database_type = 'duckdb'
        
        adapted_sql = sql_query
        db_config = self.function_maps[database_type]
        
        dprint(f"🔧 Adapting SQL for {database_type.upper()}")
        dprint(f"🔧 Original SQL: {sql_query}")
        
        # Apply date function transformations
        for pattern, replacement in db_config['date_patterns']:
            adapted_sql = re.sub(pattern, replacement, adapted_sql, flags=re.IGNORECASE)
        
        # REMOVED: STRPTIME format replacement that was adding backslashes
        # The LLM should generate the correct format directly
        
        dprint(f"🔧 Adapted SQL: {adapted_sql}")
        return adapted_sql
    
    def get_database_specific_hints(self, database_type: str) -> str:
        """Get database-specific hints for LLM prompts"""
        hints = {
            'duckdb': """
            - Use DATE_DIFF('day', start_date, end_date) for date differences
            - Use STRPTIME(date_string, '%d-%m-%Y') for date parsing
            - Use STRFTIME(date, '%Y-%m-%d') for date formatting
            - Use REGR_SLOPE(y, x) for regression slope calculations
            """,
            'postgresql': """
            - Use EXTRACT(EPOCH FROM date_column) / 86400 for Julian day
            - Use TO_CHAR(date, 'format') for date formatting
            - Use TO_DATE(string, 'format') for date parsing
            - Use REGR_SLOPE(y, x) for regression slope calculations
            """,
            'mysql': """
            - Use TO_DAYS(date) for Julian day
            - Use DATE_FORMAT(date, 'format') for date formatting
            - Use STR_TO_DATE(string, 'format') for date parsing
            - Use REGR_SLOPE(y, x) for regression slope calculations (if available)
            """,
            'sqlite': """
            - Use JULIANDAY(date) for Julian day
            - Use STRFTIME('format', date) for date formatting
            - Use STRPTIME(string, 'format') for date parsing
            """
        }
        return hints.get(database_type, hints['duckdb'])

# Global SQL function mapper instance
sql_mapper = SQLFunctionMapper()

class LLMManager:
    """Centralized LLM management to prevent multiple initializations"""
    
    _instance = None
    _llm_instances = {}
    _memory_instances = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._api_key = os.getenv("GEMINI_API_KEY")
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")
    
    def get_llm(self, model: str = "gemini-1.5-flash", temperature: float = 0.0) -> ChatGoogleGenerativeAI:
        """Get or create LLM instance with specified parameters"""
        key = f"{model}_{temperature}"
        
        if key not in self._llm_instances:
            dprint(f"🔍 LLMManager: Creating new LLM instance for {model} with temperature {temperature}")
            self._llm_instances[key] = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self._api_key,
                temperature=temperature,
                timeout=300
            )
        else:
            dprint(f"🔍 LLMManager: Reusing existing LLM instance for {model} with temperature {temperature}")
        
        return self._llm_instances[key]
    
    def get_memory(self, session_id: str = "default") -> ConversationSummaryBufferMemory:
        """Get or create memory instance for conversation context"""
        if session_id not in self._memory_instances:
            dprint(f"🔍 LLMManager: Creating new memory instance for session {session_id}")
            self._memory_instances[session_id] = ConversationSummaryBufferMemory(
                llm=self.get_llm(temperature=0.0),
                max_token_limit=5000,
                return_messages=True
            )
        else:
            dprint(f"🔍 LLMManager: Reusing existing memory instance for session {session_id}")
        
        return self._memory_instances[session_id]
    
    def cleanup(self):
        """Clean up LLM instances and memory"""
        self._llm_instances.clear()
        self._memory_instances.clear()
        dprint("🔍 LLMManager: Cleaned up all LLM instances and memory")

# Global LLM manager instance
llm_manager = LLMManager()

class SQLValidator:
    """Centralized SQL validation to eliminate duplicate code"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
    
    async def validate_and_fix_sql(self, sql_query: str, database_type: str) -> str:
        """Validate and fix SQL query using LLM"""
        try:
            dprint(f"🔍 SQLValidator: Validating SQL for {database_type}")
            dprint(f"🔍 SQLValidator: Original SQL: {sql_query[:200]}...")
            
            # If the SQL is already good, don't mess with it
            if self._is_sql_already_good(sql_query):
                dprint(f"🔍 SQLValidator: SQL is already good, returning unchanged")
                return sql_query
            
            # Get LLM instance
            llm = self.llm_manager.get_llm(temperature=0.0)
            
            verification_prompt = f"""
            You are a SQL expert. Review this SQL query and fix ONLY syntax errors if they exist.
            
            Original query: {sql_query}
            Database type: {database_type}
            
            IMPORTANT: If the SQL query is already correct and syntactically valid, return it UNCHANGED.
            Only fix actual syntax errors like:
            - Unbalanced parentheses or quotes
            - Invalid SQL keywords
            - Missing semicolons
            - Malformed column references
            
            DO NOT change:
            - Table names
            - Column names  
            - Query logic
            - WHERE conditions
            - JOIN clauses
            
            CRITICAL: Return ONLY the corrected SQL query, no explanations, no markdown, no code blocks.
            If the query is already correct, return it exactly as provided.
            """
            
            response = await llm.ainvoke(verification_prompt)
            verified_sql = response.content.strip()
            
            # Clean the verified SQL response gently
            verified_sql = self.clean_sql_response_gentle(verified_sql)
            
            dprint(f"🔍 SQLValidator: SQL validated and fixed: {verified_sql[:100]}...")
            return verified_sql
            
        except Exception as e:
            dprint(f"🔍 SQLValidator: Validation failed: {e}, using original SQL")
            return sql_query
    
    def _is_sql_already_good(self, sql_query: str) -> bool:
        """Check if SQL is already good and doesn't need fixing"""
        if not sql_query or len(sql_query) < 10:
            return False
        
        # Check for basic SQL structure
        sql_upper = sql_query.upper()
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Check for balanced parentheses (allow slight imbalance)
        open_parens = sql_query.count('(')
        close_parens = sql_query.count(')')
        if abs(open_parens - close_parens) > 1:
            return False
        
        # Check for balanced quotes
        single_quotes = sql_query.count("'")
        double_quotes = sql_query.count('"')
        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            return False
        
        return True
    
    def clean_sql_response_gentle(self, sql_response: str) -> str:
        """Clean SQL response from LLM gently - don't destroy good SQL"""
        sql_query = sql_response.strip()
        
        # Remove markdown code blocks gently
        import re
        if sql_query.startswith('```sql'):
            sql_query = sql_query[7:]
        if sql_query.startswith('```'):
            sql_query = sql_query[3:]
        if sql_query.endswith('```'):
            sql_query = sql_query[:-3]
        
        # Clean up whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query)
        sql_query = sql_query.strip()
        
        # Remove trailing semicolon
        sql_query = sql_query.rstrip(';').strip()
        
        dprint(f"🔍 SQLValidator: Gently cleaned SQL from {len(sql_response)} to {len(sql_query)} chars")
        return sql_query
    
    def clean_sql_response(self, sql_response: str) -> str:
        """Clean SQL response from LLM - legacy method, now calls gentle version"""
        return self.clean_sql_response_gentle(sql_response)

# Global SQL validator instance
sql_validator = SQLValidator(llm_manager)

class SQLErrorHandler:
    """Unified error handling for SQL operations"""
    
    @staticmethod
    def handle_error(error: Exception, operation: str, database_type: str) -> str:
        """Handle SQL errors with consistent formatting"""
        error_msg = str(error).lower()
        
        # Categorize errors
        if 'syntax' in error_msg or 'invalid' in error_msg:
            error_type = "SQL Syntax Error"
            suggestion = "Check SQL syntax and ensure proper formatting"
        elif 'connection' in error_msg or 'timeout' in error_msg:
            error_type = "Connection Error"
            suggestion = "Verify database connection and credentials"
        elif 'permission' in error_msg or 'access' in error_msg:
            error_type = "Permission Error"
            suggestion = "Check database user permissions"
        elif 'table' in error_msg or 'column' in error_msg:
            error_type = "Schema Error"
            suggestion = "Verify table and column names exist"
        elif 'memory' in error_msg or 'resource' in error_msg:
            error_type = "Resource Error"
            suggestion = "Query may be too complex, try simplifying"
        else:
            error_type = "Execution Error"
            suggestion = "Review query logic and database state"
        
        # Format error response
        error_response = {
            "error_type": error_type,
            "message": str(error),
            "operation": operation,
            "database_type": database_type,
            "suggestion": suggestion,
            "timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(error_response, indent=2)
    
    @staticmethod
    def log_error(error: Exception, context: dict):
        """Log error for monitoring and debugging"""
        # Create a copy of context to avoid modifying the original
        safe_context = context.copy()
        
        # If there's a SQL query in the context, ensure it's properly handled
        if 'query' in safe_context:
            # Store the original query for debugging but don't let JSON serialization escape it
            safe_context['query_preview'] = str(safe_context['query'])[:200] + "..." if len(str(safe_context['query'])) > 200 else str(safe_context['query'])
            # Remove the full query to prevent JSON escaping issues
            del safe_context['query']
        
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "error_type": type(error).__name__,
            "context": safe_context
        }
        dprint(f"❌ SQL Error Log: {json.dumps(error_log, indent=2)}")

# Global error handler instance
sql_error_handler = SQLErrorHandler()

class SQLQueryCache:
    """Cache system for SQL queries to avoid regeneration"""
    
    def __init__(self, max_size: int = 100):
        self._cache = {}
        self._max_size = max_size
        self._access_count = {}
        self._lock = asyncio.Lock()
    
    async def get_cached_query(self, query_hash: str) -> Optional[dict]:
        """Get cached query result"""
        async with self._lock:
            if query_hash in self._cache:
                # Update access count
                self._access_count[query_hash] = self._access_count.get(query_hash, 0) + 1
                dprint(f"🔍 SQLQueryCache: Cache hit for query hash: {query_hash[:20]}...")
                return self._cache[query_hash]
            return None
    
    async def cache_query(self, query_hash: str, sql_query: str, result: Any, database_type: str):
        """Cache a query result"""
        async with self._lock:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self._max_size:
                self._evict_least_used()
            
            cache_entry = {
                'sql_query': sql_query,
                'result': result,
                'database_type': database_type,
                'timestamp': datetime.now().isoformat(),
                'access_count': 1
            }
            
            self._cache[query_hash] = cache_entry
            self._access_count[query_hash] = 1
            dprint(f"🔍 SQLQueryCache: Cached query hash: {query_hash[:20]}...")
    
    def _evict_least_used(self):
        """Evict least recently used cache entries"""
        if not self._access_count:
            return
        
        # Find least used entry
        least_used = min(self._access_count.items(), key=lambda x: x[1])
        query_hash = least_used[0]
        
        # Remove from cache
        del self._cache[query_hash]
        del self._access_count[query_hash]
        dprint(f"🔍 SQLQueryCache: Evicted least used query: {query_hash[:20]}...")
    
    def generate_query_hash(self, query: str, database_type: str, connection_string: str) -> str:
        """Generate hash for query caching"""
        import hashlib
        query_string = f"{query}_{database_type}_{connection_string}"
        return hashlib.md5(query_string.encode()).hexdigest()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'total_cached': len(self._cache),
            'max_size': self._max_size,
            'cache_hit_ratio': self._calculate_hit_ratio()
        }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total_accesses = sum(self._access_count.values())
        if total_accesses == 0:
            return 0.0
        return len(self._cache) / total_accesses

# Global SQL query cache instance
sql_query_cache = SQLQueryCache()

class SQLTimeoutManager:
    """Manages SQL query timeouts and cancellation"""
    
    def __init__(self, default_timeout: int = 120):  # Increased default timeout for complex queries
        self.default_timeout = default_timeout
        self._active_queries = {}
        self._query_tasks = {}
    
    async def execute_with_timeout(self, query_func, *args, timeout_seconds: int = None, query_id: str = None, **kwargs):
        """Execute a query with timeout protection"""
        if timeout_seconds is None:
            timeout_seconds = self.default_timeout
        
        if query_id is None:
            query_id = f"query_{datetime.now().timestamp()}"
        
        try:
            dprint(f"🔍 SQLTimeoutManager: Executing query {query_id} with {timeout_seconds}s timeout")
            
            # Create task for execution
            task = asyncio.create_task(query_func(*args, **kwargs))
            self._query_tasks[query_id] = task
            
            # Execute with timeout
            result = await asyncio.wait_for(task, timeout=timeout_seconds)
            
            # Clean up
            if query_id in self._query_tasks:
                del self._query_tasks[query_id]
            
            dprint(f"🔍 SQLTimeoutManager: Query {query_id} completed successfully")
            return result
            
        except asyncio.TimeoutError:
            dprint(f"🔍 SQLTimeoutManager: Query {query_id} timed out after {timeout_seconds}s")
            
            # Cancel the task
            if query_id in self._query_tasks:
                self._query_tasks[query_id].cancel()
                del self._query_tasks[query_id]
            
            raise TimeoutError(f"Query execution timed out after {timeout_seconds} seconds")
        
        except Exception as e:
            dprint(f"🔍 SQLTimeoutManager: Query {query_id} failed: {e}")
            
            # Clean up
            if query_id in self._query_tasks:
                del self._query_tasks[query_id]
            
            raise
    
    def cancel_query(self, query_id: str) -> bool:
        """Cancel a running query"""
        if query_id in self._query_tasks:
            self._query_tasks[query_id].cancel()
            del self._query_tasks[query_id]
            dprint(f"🔍 SQLTimeoutManager: Cancelled query {query_id}")
            return True
        return False
    
    def get_active_queries(self) -> list:
        """Get list of active query IDs"""
        return list(self._query_tasks.keys())
    
    def get_timeout_config(self) -> dict:
        """Get timeout configuration"""
        return {
            'default_timeout': self.default_timeout,
            'active_queries': len(self._query_tasks)
        }

# Global timeout manager instance
sql_timeout_manager = SQLTimeoutManager()

class SQLConnectionPool:
    """Connection pooling for database connections"""
    
    def __init__(self, max_connections: int = 10, max_idle_time: int = 300):
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self._connections = {}
        self._connection_times = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def get_connection(self, connection_string: str, database_type: str):
        """Get a database connection from the pool"""
        pool_key = f"{database_type}_{connection_string}"
        
        async with self._lock:
            # Check if we have an available connection
            if pool_key in self._connections and self._connections[pool_key]:
                connection = self._connections[pool_key].pop()
                dprint(f"🔍 SQLConnectionPool: Reusing connection for {database_type}")
                return connection
            
            # Check if we can create a new connection
            total_connections = sum(len(conns) for conns in self._connections.values())
            if total_connections < self.max_connections:
                dprint(f"🔍 SQLConnectionPool: Creating new connection for {database_type}")
                return await self._create_connection(connection_string, database_type)
            
            # Wait for a connection to become available
            dprint(f"🔍 SQLConnectionPool: Waiting for available connection...")
            return await self._wait_for_connection(pool_key)
    
    async def return_connection(self, connection, connection_string: str, database_type: str):
        """Return a connection to the pool"""
        pool_key = f"{database_type}_{connection_string}"
        
        async with self._lock:
            if pool_key not in self._connections:
                self._connections[pool_key] = []
                self._connection_times[pool_key] = []
            
            # Check if connection is still valid
            if await self._is_connection_valid(connection, database_type):
                self._connections[pool_key].append(connection)
                self._connection_times[pool_key].append(datetime.now())
                dprint(f"🔍 SQLConnectionPool: Returned connection to pool for {database_type}")
            else:
                dprint(f"🔍 SQLConnectionPool: Connection invalid, not returning to pool for {database_type}")
                await self._close_connection(connection, database_type)
    
    async def _create_connection(self, connection_string: str, database_type: str):
        """Create a new database connection"""
        try:
            if database_type.lower() == 'duckdb':
                import duckdb
                connection = duckdb.connect(':memory:')
                # Install and load required extensions
                connection.execute("INSTALL httpfs; LOAD httpfs;")
                connection.execute("INSTALL parquet; LOAD parquet;")
                return connection
            elif database_type.lower() == 'postgresql':
                from sqlalchemy import create_engine
                engine = create_engine(connection_string)
                return engine
            elif database_type.lower() == 'mysql':
                from sqlalchemy import create_engine
                engine = create_engine(connection_string)
                return engine
            else:
                raise ValueError(f"Unsupported database type: {database_type}")
        except Exception as e:
            dprint(f"🔍 SQLConnectionPool: Failed to create connection: {e}")
            raise
    
    async def _is_connection_valid(self, connection, database_type: str) -> bool:
        """Check if a connection is still valid"""
        try:
            if database_type.lower() == 'duckdb':
                # DuckDB connections are always valid
                return True
            elif database_type.lower() in ['postgresql', 'mysql']:
                # Test connection with a simple query
                with connection.connect() as conn:
                    conn.execute("SELECT 1")
                return True
            return False
        except Exception:
            return False
    
    async def _close_connection(self, connection, database_type: str):
        """Close a database connection"""
        try:
            if hasattr(connection, 'close'):
                connection.close()
            dprint(f"🔍 SQLConnectionPool: Closed connection for {database_type}")
        except Exception as e:
            dprint(f"🔍 SQLConnectionPool: Error closing connection: {e}")
    
    async def _wait_for_connection(self, pool_key: str):
        """Wait for a connection to become available"""
        # Simple implementation - in production, you'd want a proper queue
        await asyncio.sleep(0.1)
        return await self.get_connection(pool_key.split('_', 1)[1], pool_key.split('_', 1)[0])
    
    async def cleanup_idle_connections(self):
        """Clean up idle connections"""
        async with self._lock:
            current_time = datetime.now()
            for pool_key in list(self._connections.keys()):
                if pool_key in self._connection_times:
                    # Remove connections older than max_idle_time
                    valid_connections = []
                    valid_times = []
                    
                    for i, conn_time in enumerate(self._connection_times[pool_key]):
                        if (current_time - conn_time).seconds < self.max_idle_time:
                            valid_connections.append(self._connections[pool_key][i])
                            valid_times.append(conn_time)
                    
                    self._connections[pool_key] = valid_connections
                    self._connection_times[pool_key] = valid_times
    
    def get_pool_stats(self) -> dict:
        """Get connection pool statistics"""
        total_connections = sum(len(conns) for conns in self._connections.values())
        return {
            'total_connections': total_connections,
            'max_connections': self.max_connections,
            'pool_utilization': total_connections / self.max_connections if self.max_connections > 0 else 0,
            'database_pools': {k: len(v) for k, v in self._connections.items()}
        }

# Global connection pool instance
sql_connection_pool = SQLConnectionPool()

class FallbackLoopManager:
    """Manages fallback loops and re-planning when tasks fail"""
    
    def __init__(self, max_retries: int = 2, max_replans: int = 2):
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.retry_counts = {}
        self.replan_counts = {}
        self.failure_logs = {}
    
    def should_retry_task(self, task_id: str) -> bool:
        """Check if a task should be retried"""
        current_retries = self.retry_counts.get(task_id, 0)
        return current_retries < self.max_retries
    
    def should_replan(self, execution_id: str) -> bool:
        """Check if execution should be re-planned"""
        current_replans = self.replan_counts.get(execution_id, 0)
        return current_replans < self.max_replans
    
    def record_task_failure(self, task_id: str, error: str, context: dict = None):
        """Record a task failure for analysis"""
        if task_id not in self.failure_logs:
            self.failure_logs[task_id] = []
        
        failure_info = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'context': context or {},
            'retry_count': self.retry_counts.get(task_id, 0)
        }
        self.failure_logs[task_id].append(failure_info)
        
        # Increment retry count
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1
    
    def record_execution_failure(self, execution_id: str, error: str, failed_tasks: list = None):
        """Record an execution failure for re-planning analysis"""
        if execution_id not in self.failure_logs:
            self.failure_logs[execution_id] = []
        
        failure_info = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'failed_tasks': failed_tasks or [],
            'replan_count': self.replan_counts.get(execution_id, 0)
        }
        self.failure_logs[execution_id].append(failure_info)
        
        # Increment replan count
        self.replan_counts[execution_id] = self.replan_counts.get(execution_id, 0) + 1
    
    def analyze_failures(self, task_id: str = None, execution_id: str = None) -> dict:
        """Analyze failure patterns to suggest improvements"""
        if task_id and task_id in self.failure_logs:
            failures = self.failure_logs[task_id]
            return {
                'task_id': task_id,
                'total_failures': len(failures),
                'retry_count': self.retry_counts.get(task_id, 0),
                'common_errors': self._extract_common_errors(failures),
                'suggestions': self._generate_suggestions(failures)
            }
        elif execution_id and execution_id in self.failure_logs:
            failures = self.failure_logs[execution_id]
            return {
                'execution_id': execution_id,
                'total_failures': len(failures),
                'replan_count': self.replan_counts.get(execution_id, 0),
                'failed_tasks': failures[-1].get('failed_tasks', []) if failures else [],
                'suggestions': self._generate_execution_suggestions(failures)
            }
        return {}
    
    def _extract_common_errors(self, failures: list) -> list:
        """Extract common error patterns from failures"""
        error_counts = {}
        for failure in failures:
            error = failure.get('error', '')
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Return top 3 most common errors
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _generate_suggestions(self, failures: list) -> list:
        """Generate suggestions based on failure patterns"""
        suggestions = []
        
        # Analyze error patterns
        sql_errors = [f for f in failures if 'sql' in f.get('error', '').lower()]
        data_errors = [f for f in failures if 'data' in f.get('error', '').lower()]
        connection_errors = [f for f in failures if 'connection' in f.get('error', '').lower()]
        
        if len(sql_errors) > len(failures) * 0.5:
            suggestions.append("Consider improving SQL generation prompts or adding more database-specific examples")
        
        if len(data_errors) > len(failures) * 0.5:
            suggestions.append("Data validation or preprocessing may be needed before analysis")
        
        if len(connection_errors) > len(failures) * 0.5:
            suggestions.append("Database connection issues detected - check credentials and network connectivity")
        
        if len(failures) >= 3:
            suggestions.append("Multiple failures suggest systematic issues - consider tool redesign or better error handling")
        
        return suggestions
    
    def _generate_execution_suggestions(self, failures: list) -> list:
        """Generate suggestions for execution-level failures"""
        suggestions = []
        
        if not failures:
            return suggestions
        
        latest_failure = failures[-1]
        failed_tasks = latest_failure.get('failed_tasks', [])
        
        if len(failed_tasks) > 1:
            suggestions.append("Multiple task failures suggest dependency or resource issues")
        
        if 'timeout' in latest_failure.get('error', '').lower():
            suggestions.append("Consider increasing timeout limits or breaking complex tasks into smaller ones")
        
        if 'memory' in latest_failure.get('error', '').lower():
            suggestions.append("Memory issues detected - consider data streaming or chunking approaches")
        
        return suggestions
    
    def reset_counts(self, task_id: str = None, execution_id: str = None):
        """Reset retry/replan counts for specific task or execution"""
        if task_id:
            self.retry_counts[task_id] = 0
        if execution_id:
            self.replan_counts[execution_id] = 0
    
    def get_status_summary(self) -> dict:
        """Get overall status summary of the fallback system"""
        return {
            'total_tasks_with_failures': len(self.retry_counts),
            'total_executions_with_failures': len([k for k in self.failure_logs.keys() if k.startswith('exec_')]),
            'most_failed_task': max(self.retry_counts.items(), key=lambda x: x[1]) if self.retry_counts else None,
            'most_failed_execution': max(self.replan_counts.items(), key=lambda x: x[1]) if self.replan_counts else None
        }

# Global fallback loop manager instance
fallback_manager = FallbackLoopManager()

@app.get("/fallback/status")
async def get_fallback_status():
    """Get the overall status of the fallback loop system"""
    return {
        "status": "active",
        "summary": fallback_manager.get_status_summary(),
        "config": {
            "max_retries": fallback_manager.max_retries,
            "max_replans": fallback_manager.max_replans
        }
    }

@app.get("/fallback/analysis/{task_id}")
async def get_task_failure_analysis(task_id: str):
    """Get detailed failure analysis for a specific task"""
    analysis = fallback_manager.analyze_failures(task_id=task_id)
    if not analysis:
        return {"error": f"No failure data found for task: {task_id}"}
    return analysis

@app.get("/fallback/analysis/execution/{execution_id}")
async def get_execution_failure_analysis(execution_id: str):
    """Get detailed failure analysis for a specific execution"""
    analysis = fallback_manager.analyze_failures(execution_id=execution_id)
    if not analysis:
        return {"error": f"No failure data found for execution: {execution_id}"}
    return analysis

@app.post("/fallback/reset/{task_id}")
async def reset_task_retry_count(task_id: str):
    """Reset retry count for a specific task"""
    fallback_manager.reset_counts(task_id=task_id)
    return {"message": f"Retry count reset for task: {task_id}"}

@app.post("/fallback/reset/execution/{execution_id}")
async def reset_execution_replan_count(execution_id: str):
    """Reset replan count for a specific execution"""
    fallback_manager.reset_counts(execution_id=execution_id)
    return {"message": f"Replan count reset for execution: {execution_id}"}

@app.get("/sql/mapping/status")
async def get_sql_mapping_status():
    """Get the status of the SQL function mapping system"""
    return {
        "status": "active",
        "supported_databases": list(sql_mapper.function_maps.keys()),
        "total_function_mappings": sum(len(db_config['date_patterns']) for db_config in sql_mapper.function_maps.values())
    }

@app.get("/sql/mapping/{database_type}")
async def get_database_mapping_info(database_type: str):
    """Get detailed mapping information for a specific database type"""
    if database_type not in sql_mapper.function_maps:
        return {"error": f"Unsupported database type: {database_type}"}
    
    db_config = sql_mapper.function_maps[database_type]
    return {
        "database_type": database_type,
        "date_functions": db_config['date_functions'],
        "date_patterns": db_config['date_patterns'],
        "constraints": db_config.get('constraints', {}),
        "hints": sql_mapper.get_database_specific_hints(database_type)
    }

@app.get("/sql/cache/status")
async def get_sql_cache_status():
    """Get the status of the SQL query cache system"""
    return {
        "status": "active",
        "cache_stats": sql_query_cache.get_cache_stats()
    }

@app.get("/sql/timeout/status")
async def get_sql_timeout_status():
    """Get the status of the SQL timeout management system"""
    return {
        "status": "active",
        "timeout_config": sql_timeout_manager.get_timeout_config(),
        "active_queries": sql_timeout_manager.get_active_queries()
    }

@app.get("/sql/connection-pool/status")
async def get_sql_connection_pool_status():
    """Get the status of the SQL connection pool system"""
    return {
        "status": "active",
        "pool_stats": sql_connection_pool.get_pool_stats()
    }

@app.get("/llm/status")
async def get_llm_manager_status():
    """Get the status of the centralized LLM manager"""
    return {
        "status": "active",
        "llm_instances": list(llm_manager._llm_instances.keys()),
        "total_instances": len(llm_manager._llm_instances)
    }

@app.post("/sql/timeout/cancel/{query_id}")
async def cancel_sql_query(query_id: str):
    """Cancel a running SQL query"""
    success = sql_timeout_manager.cancel_query(query_id)
    return {
        "message": f"Query {query_id} {'cancelled' if success else 'not found'}",
        "success": success
    }

@app.post("/sql/cache/clear")
async def clear_sql_cache():
    """Clear the SQL query cache"""
    sql_query_cache._cache.clear()
    sql_query_cache._access_count.clear()
    return {"message": "SQL query cache cleared successfully"}

@app.post("/sql/connection-pool/cleanup")
async def cleanup_sql_connections():
    """Clean up idle SQL connections"""
    await sql_connection_pool.cleanup_idle_connections()
    return {"message": "Idle SQL connections cleaned up successfully"}

@app.post("/sql/test")
async def test_sql_tool():
    """Test endpoint to verify SQL tool functionality"""
    try:
        # Test with a simple query
        test_query = "SELECT COUNT(*) FROM data LIMIT 1"
        test_db_type = "duckdb"
        
        # Test SQL validation
        cleaned_sql = sql_validator.clean_sql_response(test_query)
        validated_sql = await sql_validator.validate_and_fix_sql(cleaned_sql, test_db_type)
        
        return {
            "status": "success",
            "test_query": test_query,
            "cleaned_sql": cleaned_sql,
            "validated_sql": validated_sql,
            "validation_passed": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
    }

if __name__ == "__main__":
    import uvicorn
    import atexit
    
    # Register cleanup functions
    def cleanup_on_exit():
        """Cleanup resources on application exit"""
        try:
            llm_manager.cleanup()
            dprint("🔍 Cleaned up LLM manager")
        except Exception as e:
            dprint(f"⚠️ Error during LLM cleanup: {e}")
    
    atexit.register(cleanup_on_exit)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
