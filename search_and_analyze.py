import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import json
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

load_dotenv()

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

async def search_pinecone_similar(query: str, index, top_k=10, min_score=0.05) -> List[Dict[str, Any]]:
    """
    Search Pinecone with balanced results (top text + top tables).
    """
    try:
        # Search for all results
        results = await search_pinecone_similar_raw(query, index, top_k=20, min_score=min_score)
        
        if not results:
            return []
        
        print(f"🔍 Raw search returned {len(results)} results")
        
        # Separate text and table results
        text_results = [r for r in results if r.get('content_type') != 'table']
        table_results = [r for r in results if r.get('content_type') == 'table']
        
        print(f"📊 Found {len(text_results)} text results and {len(table_results)} table results")
        
        # If no tables found in search, do a separate table-only search
        if not table_results:
            print("🔍 No tables found in main search, doing table-specific search...")
            table_results = await search_pinecone_tables_only(query, index)
            print(f"📊 Table-specific search found {len(table_results)} tables")
        
        # Create balanced results: top 3 text + top 5 tables (for LLM selection)
        balanced_results = []
        
        # Add top 3 text results
        balanced_results.extend(text_results[:3])
        
        # Add top 5 table results (for LLM to choose from)
        balanced_results.extend(table_results[:5])
        
        # Sort by score for final ordering
        balanced_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        print(f"📊 Balanced results: {len(text_results[:3])} text + {len(table_results[:5])} tables")
        
        return balanced_results
        
    except Exception as e:
        print(f"❌ Error in balanced search: {str(e)}")
        return []

async def search_pinecone_tables_only(query: str, index, top_k=10) -> List[Dict[str, Any]]:
    """
    Search for tables only in Pinecone index.
    """
    try:
        results = index.search(
            namespace="scraped-content",
            query={
                "top_k": top_k,
                "inputs": {'text': query}
            }
        )
        
        if not results or 'result' not in results or 'hits' not in results['result']:
            return []
        
        matches = results['result']['hits']
        table_results = []
        
        for match in matches:
            pinecone_score = match.get('_score', 0)
            metadata = match.get('fields', {})
            
            if not metadata or metadata.get('content_type') != 'table':
                continue
            
            # Accept any table with a reasonable semantic similarity score
            if pinecone_score >= 0.05:
                table_info = {
                    "table_id": metadata.get('table_id', ''),
                    "num_rows": metadata.get('num_rows', 0),
                    "num_columns": metadata.get('num_columns', 0),
                    "columns": metadata.get('columns', [])
                }
                
                result_item = {
                    "score": pinecone_score,
                    "content": metadata.get('summary', ''),
                    "content_type": "table",
                    "table_info": table_info
                }
                
                table_results.append(result_item)
            
        # If no tables found with the original query, try a generic table search
        if not table_results:
            print("🔍 No tables found with original query, trying generic table search...")
            generic_results = index.search(
                namespace="scraped-content",
                query={
                    "top_k": 100,
                    "inputs": {'text': "data table"}
                }
            )
            
            if generic_results and 'result' in generic_results and 'hits' in generic_results['result']:
                for match in generic_results['result']['hits']:
                    pinecone_score = match.get('_score', 0)
                    metadata = match.get('fields', {})
                    
                    if not metadata or metadata.get('content_type') != 'table':
                        continue

                    # Accept any table with a reasonable score
                    if pinecone_score >= 0.1:
                        table_info = {
                            "table_id": metadata.get('table_id', ''),
                            "num_rows": metadata.get('num_rows', 0),
                            "num_columns": metadata.get('num_columns', 0),
                            "columns": metadata.get('columns', [])
                        }
                        
                        result_item = {
                            "score": pinecone_score,
                            "content": metadata.get('summary', ''),
                            "content_type": "table",
                            "table_info": table_info
                        }
                        
                        table_results.append(result_item)
        
        # Sort by semantic similarity score and return top 5
        table_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        print(f"📊 Found {len(table_results)} tables, returning top 5 by semantic similarity")
        return table_results[:5]
        
    except Exception as e:
        print(f"❌ Error in table-only search: {str(e)}")
        return []

async def advanced_search_pinecone(query: str, index, top_k=15, min_score=0.05) -> List[Dict[str, Any]]:
    """
    Advanced multi-stage search with query expansion and semantic understanding.
    """
    try:
        # Stage 1: Direct semantic search
        print(f"🔍 Stage 1: Direct semantic search for '{query}'")
        direct_results = await _perform_semantic_search(query, index, top_k, min_score)
        
        # Stage 2: Query expansion and synonym search
        print(f"🔍 Stage 2: Query expansion for '{query}'")
        expanded_results = await _perform_expanded_search(query, index, top_k, min_score)
        
        # Stage 3: Context-aware search
        print(f"🔍 Stage 3: Context-aware search for '{query}'")
        context_results = await _perform_context_search(query, index, top_k, min_score)
        
        # Combine and deduplicate results
        all_results = direct_results + expanded_results + context_results
        deduplicated_results = _deduplicate_results(all_results)
            
        # Sort by relevance score
        deduplicated_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        print(f"📊 Advanced search found {len(deduplicated_results)} unique results")
        return deduplicated_results[:top_k]
                
    except Exception as e:
        print(f"❌ Error in advanced search: {str(e)}")
        # Fallback to basic search
        return await search_pinecone_similar_raw(query, index, top_k, min_score)

async def _perform_semantic_search(query: str, index, top_k: int, min_score: float) -> List[Dict[str, Any]]:
    """Perform direct semantic search."""
    try:
        results = index.search(
            namespace="scraped-content",
            query={
                "top_k": top_k,
                "inputs": {'text': query}
            }
        )
        
        return _process_search_results(results, min_score)
    except Exception as e:
        print(f"❌ Error in semantic search: {str(e)}")
        return []

async def _perform_expanded_search(query: str, index, top_k: int, min_score: float) -> List[Dict[str, Any]]:
    """Perform search with query expansion."""
    try:
        # Generate expanded queries
        expanded_queries = _generate_expanded_queries(query)
        
        all_results = []
        for expanded_query in expanded_queries:
            results = index.search(
                namespace="scraped-content",
                query={
                    "top_k": top_k // 2,
                    "inputs": {'text': expanded_query}
                }
            )
            
            expanded_results = _process_search_results(results, min_score)
            all_results.extend(expanded_results)
        
        return all_results
    except Exception as e:
        print(f"❌ Error in expanded search: {str(e)}")
        return []

async def _perform_context_search(query: str, index, top_k: int, min_score: float) -> List[Dict[str, Any]]:
    """Perform context-aware search."""
    try:
        # Extract key terms and concepts
        key_terms = _extract_key_terms(query)
        
        all_results = []
        for term in key_terms:
            results = index.search(
                namespace="scraped-content",
                query={
                    "top_k": top_k // 3,
                    "inputs": {'text': term}
                }
            )
            
            context_results = _process_search_results(results, min_score * 0.8)  # Lower threshold for context
            all_results.extend(context_results)
        
        return all_results
    except Exception as e:
        print(f"❌ Error in context search: {str(e)}")
        return []

def _generate_expanded_queries(query: str) -> List[str]:
    """Generate expanded queries using synonyms and related terms."""
    expanded_queries = [query]
    
    # Common synonyms and related terms - GENERALIZED
    synonyms = {
        'payment': ['payment', 'fee', 'cost', 'charge'],
        'coverage': ['coverage', 'benefits', 'protection'],
        'policy': ['policy', 'plan', 'agreement'],
        'treatment': ['treatment', 'medical', 'healthcare', 'care']
    }
    
    # Check for known terms and add synonyms
    query_lower = query.lower()
    for term, related_terms in synonyms.items():
        if term in query_lower:
            for related_term in related_terms:
                if related_term not in query_lower:
                    expanded_query = query.replace(term, related_term)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
    
    return expanded_queries

def _extract_key_terms(query: str) -> List[str]:
    """Extract key terms from the query."""
    # Simple key term extraction
    words = query.lower().split()
    
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose'}
    
    key_terms = [word for word in words if word not in stop_words and len(word) > 2]
    
    return key_terms

def _process_search_results(results: Dict, min_score: float) -> List[Dict[str, Any]]:
    """Process and format search results."""
    processed_results = []
    
    if not results or 'result' not in results or 'hits' not in results['result']:
        return processed_results
    
    matches = results['result']['hits']
    
    for match in matches:
        pinecone_score = match.get('_score', 0)
        metadata = match.get('fields', {})
        
        if pinecone_score >= min_score and metadata:
            # Extract content properly
            content = metadata.get('content', '')
            if not content:
                # Try alternative content fields
                content = metadata.get('text', '')
            if not content:
                # For tables, use summary
                if metadata.get('content_type') == 'table':
                    content = metadata.get('summary', '')
            
            result_item = {
                "score": pinecone_score,
                "content": content,
                "content_type": metadata.get('content_type', 'text'),
                "metadata": metadata
            }
            
            # Add table-specific info if available
            if metadata.get('content_type') == 'table':
                result_item["table_info"] = {
                    "table_id": metadata.get('table_id', ''),
                    "num_rows": metadata.get('num_rows', 0),
                    "num_columns": metadata.get('num_columns', 0),
                    "columns": metadata.get('columns', [])
                }
            
            processed_results.append(result_item)
    
    return processed_results

def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate results based on content similarity."""
    seen_contents = set()
    deduplicated = []
    
    for result in results:
        content = result.get('content', '')[:100]  # Use first 100 chars for deduplication
        content_hash = hash(content)
        
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            deduplicated.append(result)
    
    return deduplicated

async def search_pinecone_similar_raw(query: str, index, top_k=10, min_score=0.05) -> List[Dict[str, Any]]:
    """
    Raw Pinecone search function (original implementation).
    """
    try:
        results = index.search(
            namespace="scraped-content",
            query={
                "top_k": top_k,
                "inputs": {'text': query}
            }
        )
        
        if not results or 'result' not in results or 'hits' not in results['result']:
            return []
        
        matches = results['result']['hits']
        processed_results = []
        
        for match in matches:
            pinecone_score = match.get('_score', 0)
            metadata = match.get('fields', {})
            
            if not metadata:
                continue

            keywords = metadata.get('keywords', '').split(', ')
            
            # Handle both text and table content
            if metadata.get('content_type') == 'table':
                content = metadata.get('summary', '')  # Use summary for tables
                table_info = {
                    "table_id": metadata.get('table_id', ''),
                    "num_rows": metadata.get('num_rows', 0),
                    "num_columns": metadata.get('num_columns', 0),
                    "columns": metadata.get('columns', [])
                }
            else:
                # For text content, the content is stored in the 'text' field by Pinecone
                # but we need to get it from the metadata or the match itself
                content = metadata.get('content', '')  # Try metadata first
                if not content:
                    # If not in metadata, try to get from the match
                    content = match.get('text', '')  # Pinecone might store it here
                if not content:
                    # Last resort - try to reconstruct from other fields
                    content = metadata.get('text', '')  # Some implementations store it here
                
                table_info = None

            # Calculate keyword score
            keyword_score = 0
            query_terms = set(query.lower().split())
            
            keyword_match_count = sum(1 for keyword in keywords if keyword in query_terms)
            if keyword_match_count > 0:
                keyword_score += 0.1 * keyword_match_count

            # Hybrid score
            final_score = (pinecone_score * 0.7) + (keyword_score * 0.3)

            if final_score >= min_score:
                result_item = {
                    "score": final_score,
                    "content": content,
                    "content_type": metadata.get('content_type', 'text'),
                }
                
                # Add table-specific information
                if table_info:
                    result_item["table_info"] = table_info
                
                processed_results.append(result_item)
                
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        return processed_results
        
    except Exception as e:
        print(f"❌ Error in raw search: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

async def analyze_with_gemini(query: str, search_results: List[Dict[str, Any]], scraped_content: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze search results with Gemini LLM.
    """
    try:
        # If no search results found, try direct content search
        if not search_results and scraped_content:
            # Fallback to direct content search if Pinecone yields no results
            print("No vector search results found. Falling back to direct content search...")
            
            all_content = "".join([page.get('content', '') for page in scraped_content])
            
            if query.lower() in all_content.lower():
                 return {
                    "answer": "I found a direct match for your query in the text, but was unable to perform a full analysis. The content is available for review.",
                    "sources": [{"title": "Full Scraped Content", "score": 1.0}],
                    "confidence": "medium"
                }

        # If still no search results found, return empty result
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the scraped content to answer your question.",
                "sources": [],
                "confidence": "low"
            }
        
        # Format the search results for Gemini - limit to top 3 most relevant results
        context_parts = []
        table_found = False
        table_ids = []
        
        for i, result in enumerate(search_results[:3]):  # Only top 3 results
            if result.get('content_type') == 'table':
                table_found = True
                table_info = result.get('table_info', {})
                table_ids.append(table_info.get('table_id', ''))
                
                # Load a small preview from the actual table file for context
                try:
                    table_path = table_info.get('table_path', '')
                    if not table_path:
                        table_path = f"temp_files/{table_info.get('table_id', '')}"
                    
                    import pandas as pd
                    df = pd.read_csv(table_path)
                    first_row_example = df.head(1).to_dict('records')[0] if len(df) > 0 else {}
                    first_row_str = ", ".join([f"{k}: {v}" for k, v in first_row_example.items()])
                    
                    # Create a small preview (first 3 rows)
                    preview_markdown = df.head(3).to_markdown(index=False) if len(df) > 0 else "No data"
                except Exception as e:
                    first_row_str = "Could not load first row example"
                    preview_markdown = "Could not load table preview"
                
                # For tables, include summary, structure info, and first row example
                context_part = f"""Source {i+1} - TABLE (Relevance Score: {result['score']:.3f}):
Summary: {result['content'][:400]}...
Table Structure: {table_info.get('num_rows', 0)} rows, {table_info.get('num_columns', 0)} columns
Columns: {', '.join([str(col) for col in table_info.get('columns', [])[:10]])}
First Row Example: {first_row_str}
Table ID: {table_info.get('table_id', 'N/A')}

Table Preview (first 3 rows):
```markdown
{preview_markdown}
```"""
            else:
                # Regular text content - increase limit to show more content (was 2000, now 4000)
                content_preview = result['content'][:4000] + "..." if len(result['content']) > 4000 else result['content']
            context_part = f"""Source {i+1} (Relevance Score: {result['score']:.3f}):
{content_preview}"""
            
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for handling tables
        if table_found:
            # Filter for high relevance tables only
            high_relevance_tables = [tid for i, tid in enumerate(table_ids) if search_results[i]['score'] > 0.3]
            
            prompt = f"""You are an expert data analyst. The user has asked a question and I've found relevant information including structured data tables. 

IMPORTANT: If the user's question requires specific calculations, filtering, or data analysis on the table data, you should:
1. First indicate that you found relevant table data with HIGH relevance scores
2. Mention the high-scoring table ID(s): {', '.join(high_relevance_tables)}
3. Suggest that for precise calculations, the DataframeAnalysisTool should be used with the table ID
4. Still provide what insights you can from the visible data and first row examples

User Query: {query}

Relevant Information (Only showing tables with relevance score > 0.3):
{context}

Please provide a comprehensive answer. If the question requires precise calculations or filtering of the table data, mention that more detailed analysis can be performed using the DataframeAnalysisTool with the table ID(s): {', '.join(high_relevance_tables)}"""
        else:
            prompt = f"""You are an expert analyst. Answer the user's question based on the provided context. Synthesize information from all sources to provide a complete, accurate answer.

IMPORTANT: When the user asks for specific information, look for EXACT details in the context and provide them precisely. If you find specific information that directly answers the question, state it clearly and directly.

CRITICAL: If you find information that answers the user's question, even if it uses different terminology than what the user asked for, PROVIDE THAT INFORMATION. Do not say "the information doesn't exist" if you found relevant information that answers the question.

SPECIFIC INSTRUCTIONS: If you find a definition or specific details that answer the user's question, quote the exact text and provide the precise answer.

User Query: {query}

Context:
{context}

Please provide a comprehensive answer based on the available information. If you find specific details that directly answer the question, make sure to include them prominently in your answer."""

        # Generate response with Gemini
        # Initialize Gemini model inside the function to avoid event loop issues
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1000
            }
        )
        
        if response and hasattr(response, 'text'):
            answer = response.text.strip()
            
            # Extract sources used
            sources = []
            for result in search_results[:3]:  # Top 3 sources
                sources.append({
                    "score": result["score"],
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                })
            
            # Determine confidence based on scores
            avg_score = sum(r["score"] for r in search_results) / len(search_results)
            if avg_score > 0.7:
                confidence = "high"
            elif avg_score > 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "avg_score": avg_score,
                "total_sources": len(search_results)
            }
        
        return {
            "answer": "I encountered an error while generating the analysis.",
            "sources": [],
            "confidence": "low"
        }
        
    except Exception as e:
        print(f"Error analyzing with Gemini: {str(e)}")
        return {
            "answer": f"I encountered an error while analyzing the content: {str(e)}",
            "sources": [],
            "confidence": "low"
        }

async def triage_user_query(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phase 1: Triage the user query to determine the next action.
    This function analyzes search results and decides whether to:
    - ANALYZE_TABLE: Use DataframeAnalysisTool for data-driven questions
    - SYNTHESIZE_TEXT: Use text-based analysis for general questions
    
    For table analysis, it presents multiple table candidates and lets the LLM choose the best one.
    """
    try:
        # Separate text and table results
        text_results = [r for r in search_results if r.get('content_type') != 'table']
        table_results = [r for r in search_results if r.get('content_type') == 'table']
        
        # Create a balanced context for the triage LLM
        triage_context = []
        
        # Add top 3 text results
        for i, result in enumerate(text_results[:3]):
            triage_context.append({
                "type": "text",
                "score": result.get('score', 0),
                "content": result.get('content', '')[:500]  # Limit length
            })
        
        # Add ALL available table results (up to 4) for LLM selection
        available_table_ids = []
        for i, result in enumerate(table_results[:4]):
            table_info = result.get('table_info', {})
            table_id = table_info.get('table_id', '')
            available_table_ids.append(table_id)
            
            # Get actual table content for better evaluation
            table_content = ""
            try:
                import pandas as pd
                table_path = f"temp_files/{table_id}"
                if os.path.exists(table_path):
                    df = pd.read_csv(table_path)
                    # Get column names and sample data
                    columns = list(df.columns)
                    sample_data = df.head(3).to_string()
                    table_content = f"Columns: {columns}\nSample data:\n{sample_data}"
                else:
                    table_content = f"Columns: {table_info.get('columns', [])}\nSummary: {result.get('content', '')}"
            except Exception as e:
                table_content = f"Columns: {table_info.get('columns', [])}\nSummary: {result.get('content', '')}"
            
            triage_context.append({
                "type": "table",
                "score": result.get('score', 0),
                "table_id": table_id,
                "summary": result.get('content', ''),
                "columns": table_info.get('columns', []),
                "num_rows": table_info.get('num_rows', 0),
                "num_columns": table_info.get('num_columns', 0),
                "table_content": table_content,  # Add actual table content
                "caption": table_info.get('caption', ''),  # Add table caption
                "associated_header": table_info.get('associated_header', '')  # Add associated header
            })
        
        # Create the enhanced triage prompt
        triage_prompt = f"""You are a workflow routing expert. Your job is to determine if a user's question can be answered by analyzing the provided text, or if it requires running programmatic code on a structured data table.

IMPORTANT: Look for questions that require:
- Sorting data (e.g., "earliest", "latest", "highest", "lowest")
- Filtering by specific values (e.g., "over $1.5bn", "before 2000", "more than 100")
- Counting or aggregating data (e.g., "how many", "total", "average")
- Complex data analysis that cannot be answered from text alone

User Query: "{query}"

Available Information:
{json.dumps(triage_context, indent=2)}

CRITICAL INSTRUCTIONS FOR TABLE SELECTION:
If you determine that table analysis is needed, you MUST carefully examine ALL available tables and choose the SINGLE BEST table that is most likely to contain the answer to the user's question. 

For each table, examine in order of importance:
1. **Associated Header**: This is the MOST IMPORTANT piece of context
   - The header (h1, h2, h3, etc.) that comes before the table describes its true purpose
   - Look for headers that directly match the query topic
   - Headers are more reliable than captions or semantic scores for understanding table purpose

2. **Table Caption**: Secondary context about the table
   - Look for captions that match the query topic
   - Captions provide additional context about the table's purpose

3. **Column Names**: Look for exact matches or close matches to what the query needs

4. **Sample Data**: Check if the data actually contains the type of information needed

5. **Data Quality**: Will the data actually answer the question?

6. **Semantic Score**: Use as a tiebreaker, but prioritize actual content over scores

CRITICAL: The associated header is the most reliable indicator of a table's true purpose. If a table has a header that directly matches your query topic, prioritize it over other factors.

AVAILABLE TABLE IDs: {available_table_ids}
CRITICAL: You MUST select a table_id from this exact list above. Do not make up or guess table_ids.

Respond with a JSON object in this exact format:
{{
    "next_action": "ANALYZE_TABLE" | "SYNTHESIZE_TEXT",
    "reasoning": "Brief explanation of your decision",
    "table_id": "table_xxx.csv" (only if next_action is ANALYZE_TABLE - MUST be from available tables),
    "analysis_query": "Specific question for the data analysis tool" (only if next_action is ANALYZE_TABLE)
}}

Examples:
- For "Which is the earliest film that grossed over $1.5bn?" → ANALYZE_TABLE with table containing film data if present
- For "What is the history of cinema?" → SYNTHESIZE_TEXT
- For "How many films were released in 2020?" → ANALYZE_TABLE with table containing film data if present
- For "Tell me about James Cameron" → SYNTHESIZE_TEXT

CRITICAL: If you choose ANALYZE_TABLE, you MUST provide a valid table_id from the available information above."""

        # Initialize Gemini for triage
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = await model.generate_content_async(triage_prompt)
        triage_result = response.text.strip()
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            json_start = triage_result.find('{')
            json_end = triage_result.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                triage_json = json.loads(triage_result[json_start:json_end])
                return triage_json
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"⚠️ Error parsing triage result: {str(e)}")
            print(f"Raw response: {triage_result}")
            # Fallback to text synthesis
            return {
                "next_action": "SYNTHESIZE_TEXT",
                "reasoning": "Failed to parse triage result, falling back to text analysis"
            }
            
    except Exception as e:
        print(f"❌ Error in triage: {str(e)}")
        return {
            "next_action": "SYNTHESIZE_TEXT", 
            "reasoning": f"Triage failed: {str(e)}"
        }

async def execute_table_analysis(query: str, triage_result: Dict[str, Any], search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute table analysis using LangChain's Pandas DataFrame Agent.
    """
    try:
        table_id = triage_result.get('table_id')
        analysis_query = triage_result.get('analysis_query', query)
    
        if not table_id or table_id == 'unknown':
            # Handle case where no specific table was selected by triage
            return await execute_text_synthesis(query, search_results)
        
        table_path = f"temp_files/{table_id}"
        if not os.path.exists(table_path):
            return {"analysis": {"answer": f"Error: The selected table file '{table_id}' was not found.", "confidence": "low"}}

        df = pd.read_csv(table_path)

        # Initialize the LangChain LLM
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.0)

        # Create and run the Pandas DataFrame Agent
        pandas_agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            agent_executor_kwargs={"handle_parsing_errors": True},
            verbose=True,
            allow_dangerous_code=True
        )
        
        response = await pandas_agent.ainvoke(analysis_query)
        table_result = response.get("output", "Analysis completed, but no specific output was generated.")
        
        # The LangChain agent gives the final answer, but we can still synthesize if needed
        return await synthesize_with_table_result(query, table_result, search_results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"analysis": {"answer": f"An error occurred during table analysis with LangChain: {str(e)}", "confidence": "low"}}

async def execute_text_synthesis(query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute traditional text-based analysis using Gemini.
    """
    try:
        print("📝 Performing text-based analysis...")
        
        # Use the existing analyze_with_gemini function
        analysis_result = await analyze_with_gemini(query, search_results)
        
        # analyze_with_gemini already returns a dict with answer, sources, confidence, etc.
        # We need to wrap it in the expected structure
        return {
            "analysis": analysis_result,
            "method": "text_synthesis"
        }
        
    except Exception as e:
        print(f"❌ Error in text synthesis: {str(e)}")
        return {
            "analysis": {
                "answer": f"An error occurred during text analysis: {str(e)}",
                "confidence": "low",
                "reasoning": f"Error: {str(e)}"
            },
            "method": "text_synthesis_error"
        }

async def synthesize_with_table_result(query: str, table_result: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synthesize table analysis result with text context for final answer.
    """
    try:
        # Get top text results for context
        text_results = [r for r in search_results if r.get('content_type') != 'table'][:3]
        text_context = "\n\n".join([r.get('content', '') for r in text_results])
        
        # Create synthesis prompt
        synthesis_prompt = f"""You are an expert data analyst. You have performed a programmatic analysis on a table and now need to synthesize this result with additional text context to provide a comprehensive, user-friendly answer.

    User's Original Question: "{query}"

    Programmatic Analysis Result:
    {table_result}

    Additional Text Context:
    {text_context}

    Your task is to:
    1. Understand the programmatic result (it may be raw data, JSON, or a simple answer)
    2. Combine it with the text context to provide background and explanation
    3. Present a clear, natural language answer that directly addresses the user's question
    4. If the programmatic result is unclear or seems incomplete, mention this

    Provide a comprehensive answer that synthesizes both the data analysis and the contextual information:"""

        # Initialize Gemini for synthesis
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = await model.generate_content_async(synthesis_prompt)
        final_answer = response.text.strip()
    
        return {
            "analysis": {
                "answer": final_answer,
                "confidence": "high",
                "reasoning": "Combined table analysis with text context",
                "table_analysis_result": table_result[:500] + "..." if len(table_result) > 500 else table_result
            },
            "method": "table_analysis_with_synthesis"
        }
        
    except Exception as e:
        print(f"❌ Error in synthesis: {str(e)}")
        return {
            "analysis": {
                "answer": f"Table analysis completed but synthesis failed: {str(e)}. Raw result: {table_result}",
                "confidence": "medium",
                "reasoning": f"Synthesis error: {str(e)}"
            },
            "method": "table_analysis_synthesis_error"
        }

async def search_and_analyze(query: str, index, scraped_content=None) -> Dict[str, Any]:
    """
    Main function to search and analyze content using advanced search strategies.
    """
    try:
        print(f"🔍 Starting advanced search and analysis for: '{query}'")
        
        # Use advanced search instead of basic search
        search_results = await advanced_search_pinecone(query, index, top_k=15, min_score=0.05)
        
        if not search_results:
            print("⚠️ No search results found, trying fallback search...")
            # Fallback to basic search
            search_results = await search_pinecone_similar_raw(query, index, top_k=10, min_score=0.05)
        
        if not search_results:
            return {
                "analysis": {
                    "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing your query or check if the content has been properly indexed.",
                    "confidence": "low",
                    "reasoning": "No search results found"
                },
                "method": "no_results"
            }
        
        print(f"📊 Found {len(search_results)} search results")
        
        # Triage the query to decide between table analysis and text synthesis
        triage_result = await triage_user_query(query, search_results)
        
        if triage_result.get('next_action') == 'ANALYZE_TABLE':
            print("📊 Executing table analysis...")
            return await execute_table_analysis(query, triage_result, search_results)
        else:
            print("📝 Executing text synthesis...")
            return await execute_text_synthesis(query, search_results)
            
    except Exception as e:
        print(f"❌ Error in search_and_analyze: {str(e)}")
        return {
            "analysis": {
                "answer": f"An error occurred while processing your request: {str(e)}",
                "confidence": "low",
                "reasoning": f"Error: {str(e)}"
            },
            "method": "error"
    }

if __name__ == "__main__":
    # Test the search and analyze functionality
    from pinecone import Pinecone
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "data-analyst-agent-embedder"
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    test_query = "What are the ethical problems or dangers associated with developing advanced AI?"
    result = search_and_analyze(test_query, index)
    
    print("\n" + "="*50)
    print("ANALYSIS RESULT")
    print("="*50)
    print(f"Answer: {result['analysis']['answer']}")
    print(f"Confidence: {result['analysis']['confidence']}")
    print(f"Average Score: {result['analysis'].get('avg_score', 'N/A')}")
    print(f"Sources Used: {result['analysis']['total_sources']}") 
    print("="*50)
    print("\nSOURCES:")
    for i, source in enumerate(result['analysis']['sources']):
        print(f"  Source {i+1}: Score {source['score']:.3f}")
        print(f"    Content: {source['content_preview']}")
        print("-" * 20) 