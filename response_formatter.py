#!/usr/bin/env python3
"""
Response Formatter for Data Analyst Agent
Standardizes response formatting for consistent JSON array output
"""

import json
import re
from typing import Any, Optional, List
import google.generativeai as genai
import os

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ResponseFormatter:
    """Standardizes response formatting for consistent JSON array output"""
    
    @staticmethod
    def _encode_image_paths_in_text(text: str) -> str:
        """Replace IMAGE_PATH references in free text with base64 data URIs."""
        try:
            import base64
            import os as _os
            import re as _re
            def _repl(match: re.Match) -> str:
                path = match.group(1).strip()
                try:
                    if _os.path.exists(path) and _os.path.isfile(path):
                        with open(path, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode()
                        return f"data:image/png;base64,{encoded}"
                except Exception:
                    return match.group(0)
                return match.group(0)
            return _re.sub(r"IMAGE_PATH:\s*([^\s`\"]+)", _repl, text)
        except Exception:
            return text

    @staticmethod
    def _inline_image_paths(result: Any) -> Any:
        """Inline base64 images for any IMAGE_PATH occurrences in strings or JSON lists/objects."""
        try:
            if isinstance(result, str):
                # If this is a JSON array/object as string, try to parse and process answers
                stripped = result.strip()
                if (stripped.startswith('[') and stripped.endswith(']')) or (stripped.startswith('{') and stripped.endswith('}')):
                    try:
                        parsed = json.loads(result)
                        inlined = ResponseFormatter._inline_image_paths(parsed)
                        return json.dumps(inlined)
                    except Exception:
                        # Fallback to regex replace on raw text
                        return ResponseFormatter._encode_image_paths_in_text(result)
                # Plain text
                return ResponseFormatter._encode_image_paths_in_text(result)
            elif isinstance(result, list):
                new_list = []
                for item in result:
                    if isinstance(item, dict) and 'answer' in item and isinstance(item['answer'], str):
                        item_copy = dict(item)
                        item_copy['answer'] = ResponseFormatter._encode_image_paths_in_text(item_copy['answer'])
                        new_list.append(item_copy)
                    elif isinstance(item, str):
                        new_list.append(ResponseFormatter._encode_image_paths_in_text(item))
                    else:
                        new_list.append(item)
                return new_list
            elif isinstance(result, dict):
                # Common schema: { "questions": [ {question, answer}, ... ] }
                new_obj = dict(result)
                if 'questions' in new_obj and isinstance(new_obj['questions'], list):
                    new_obj['questions'] = ResponseFormatter._inline_image_paths(new_obj['questions'])
                if 'answer' in new_obj and isinstance(new_obj['answer'], str):
                    new_obj['answer'] = ResponseFormatter._encode_image_paths_in_text(new_obj['answer'])
                return new_obj
            return result
        except Exception:
            return result

    @staticmethod
    async def _get_llm_response(prompt: str) -> str:
        """Gets response from the LLM."""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return ""

    @staticmethod
    async def format_final_answer(result: Any, user_query: str = None, custom_format: str = None) -> str:
        """
        Format the final answer according to the specified JSON array format.
        
        Expected format: [q1 answer, q2 answer, q3 answer, ... qn answer]
        
        If the txt file specifies a different format, that overrides the default.
        """
        try:
            # Always inline images before any further formatting
            result = ResponseFormatter._inline_image_paths(result)

            # If a custom format is specified, use it instead of the default
            if custom_format:
                formatted = await ResponseFormatter._format_with_custom_format(result, custom_format, user_query)
                # Final pass to inline any IMAGE_PATH the LLM may have reproduced
                formatted = ResponseFormatter._inline_image_paths(formatted)
                return formatted
            
            # Check if result is already in the correct format
            if isinstance(result, str):
                # Try to parse as JSON first
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        # Already a list - return as is without re-encoding
                        return result
                except json.JSONDecodeError:
                    pass
                
                # Check if this looks like it's already been processed by the LLM
                # Look for patterns that suggest the LLM already formatted it correctly
                if (result.strip().startswith('[') and result.strip().endswith(']') and 
                    '"' in result and ',' in result):
                    # This looks like a properly formatted JSON array from the LLM
                    # Just return it as-is - DON'T PROCESS IT FURTHER
                    return result
                
                # Check if this is a multi-question response that needs formatting
                if user_query and ResponseFormatter._is_multi_question_query(user_query):
                    # Use the helper method to extract answers from messy responses
                    extracted_answers = ResponseFormatter._extract_answers_from_messy_response(result)
                    if extracted_answers:
                        return json.dumps(extracted_answers)
                    else:
                        return ResponseFormatter._format_multi_question_response(result, user_query)
                else:
                    # Single question - wrap in array
                    return json.dumps([result])
            
            elif isinstance(result, dict):
                # Handle dictionary results
                if "error" in result:
                    return json.dumps([f"Error: {result['error']}"])
                elif "answer" in result:
                    answer = result["answer"]
                    # Check if answer is already formatted
                    if isinstance(answer, str):
                        try:
                            parsed = json.loads(answer)
                            if isinstance(parsed, list):
                                return answer  # Already formatted
                        except json.JSONDecodeError:
                            pass
                    
                    if user_query and ResponseFormatter._is_multi_question_query(user_query):
                        return ResponseFormatter._format_multi_question_response(answer, user_query)
                    else:
                        return json.dumps([answer])
                else:
                    return json.dumps([str(result)])
            
            elif isinstance(result, list):
                # Already a list - ensure it's properly formatted
                return json.dumps(result)
            
            else:
                # Other types - convert to string and wrap in array
                return json.dumps([str(result)])
                
        except Exception as e:
            # Fallback to error format
            return json.dumps([f"Error formatting response: {str(e)}"])
    
    @staticmethod
    def _is_multi_question_query(query: str) -> bool:
        """Check if the query contains multiple numbered questions"""
        if not query:
            return False
        
        # Look for numbered questions (1., 2., 3., etc.)
        question_pattern = r'\d+\.\s*[^?\n]+[?\n]'
        questions = re.findall(question_pattern, query)
        return len(questions) > 1
    
    @staticmethod
    def _format_multi_question_response(response: str, user_query: str) -> str:
        """Format multi-question response into JSON array"""
        try:
            # First, try to clean up the response by removing problematic characters
            cleaned_response = response.replace('\\"', '"').replace('\\\\', '\\')
            
            # Try to extract numbered answers from the response
            answer_pattern = r'(\d+)\.\s*([^?\n]+)'
            answers = re.findall(answer_pattern, cleaned_response)
            
            if answers:
                # Sort by question number and extract answers
                sorted_answers = sorted(answers, key=lambda x: int(x[0]))
                formatted_answers = [answer[1].strip() for answer in sorted_answers]
                return json.dumps(formatted_answers)
            
            # If no numbered answers found, try to split by common delimiters
            if 'Answer:' in cleaned_response:
                # Split by "Answer:" and clean up
                parts = cleaned_response.split('Answer:')
                if len(parts) > 1:
                    answers = [part.strip() for part in parts[1:] if part.strip()]
                    return json.dumps(answers)
            
            # Try to extract from the original query structure and create placeholder answers
            if user_query:
                question_pattern = r'\d+\.\s*([^?\n]+[?\n])'
                questions = re.findall(question_pattern, user_query)
                if questions:
                    # Try to extract answers from the messy response using the new helper
                    messy_answers = ResponseFormatter._extract_answers_from_messy_response(cleaned_response)
                    
                    if messy_answers:
                        return json.dumps(messy_answers)
                    
                    # If still no luck, create placeholder answers based on questions
                    placeholder_answers = [f"Answer to: {q.strip()}" for q in questions]
                    return json.dumps(placeholder_answers)
            
            # Final fallback - try to split by commas and clean up
            if ',' in cleaned_response:
                parts = cleaned_response.split(',')
                cleaned_parts = []
                for part in parts:
                    cleaned_part = part.strip().strip('"').strip()
                    if cleaned_part and not cleaned_part.startswith('\\'):
                        cleaned_parts.append(cleaned_part)
                
                if cleaned_parts:
                    return json.dumps(cleaned_parts)
            
            # Ultimate fallback - wrap the entire response in an array
            return json.dumps([cleaned_response])
            
        except Exception as e:
            # Fallback to wrapping response in array
            return json.dumps([response])
    
    @staticmethod
    def _extract_answers_from_messy_response(response: str) -> List[str]:
        """Extract individual answers from a messy response with mixed formatting"""
        try:
            # Simple approach: split by "1. ", "2. ", etc. and clean up
            import re
            
            # Split by numbered patterns
            parts = re.split(r'\d+\.\s*', response)
            
            if len(parts) > 1:
                answers = []
                for part in parts[1:]:  # Skip the first empty part
                    if part.strip():
                        # Clean up the part - remove all quotes, commas, and extra whitespace
                        cleaned = part.strip()
                        # Remove all quotes and commas
                        cleaned = cleaned.replace('"', '').replace(',', '').strip()
                        # Remove any remaining trailing punctuation
                        cleaned = re.sub(r'[^\w\s\-\.]+$', '', cleaned).strip()
                        if cleaned:
                            answers.append(cleaned)
                
                if answers:
                    return answers
            
            # Alternative: try to extract content between quotes
            if '"' in response:
                quote_pattern = r'"([^"]*)"'
                quoted_parts = re.findall(quote_pattern, response)
                
                if quoted_parts:
                    cleaned_parts = []
                    for part in quoted_parts:
                        cleaned_part = part.strip()
                        if cleaned_part and not cleaned_part.startswith('\\'):
                            cleaned_parts.append(cleaned_part)
                    
                    if cleaned_parts:
                        return cleaned_parts
            
            # Last resort: try to find numbered answers with a different pattern
            numbered_pattern = r'(\d+)\.\s*([^"]+?)(?=\d+\.|$)'
            numbered_matches = re.findall(numbered_pattern, response)
            
            if numbered_matches:
                sorted_numbered = sorted(numbered_matches, key=lambda x: int(x[0]))
                numbered_answers = []
                
                for question_num, answer in sorted_numbered:
                    cleaned_answer = answer.strip().strip('",').strip()
                    if cleaned_answer:
                        numbered_answers.append(cleaned_answer)
                
                if numbered_answers:
                    return numbered_answers
            
            return []
            
        except Exception as e:
            return []
    
    @staticmethod
    async def _format_with_custom_format(result: Any, custom_format: str, original_query: str) -> str:
        """Format the result according to a custom format specification using an LLM."""
        try:
            prompt = f"""
            Given the following result and the original user query, please format the output according to the custom format specification.

            **Result:**
            {result}

            **Original User Query:**
            {original_query}

            **Custom Format Specification:**
            {custom_format}

            **Formatted Output:**
            """
            
            formatted_response = await ResponseFormatter._get_llm_response(prompt)
            return formatted_response.strip()
                
        except Exception as e:
            print(f"⚠️ Error applying custom format '{custom_format}': {e}")
            # Fallback to default JSON array format on error
            return json.dumps(result) if isinstance(result, list) else json.dumps([str(result)])
    
    @staticmethod
    async def extract_answer_format_from_file(file_content: str) -> Optional[str]:
        """
        Use an LLM to extract any specified answer format from the file content.
        """
        if not file_content:
            return None
        
        prompt = f"""
        Analyze the following text and determine the desired output format for the answer. 
        If a specific format like a JSON schema, CSV, or a list is described, extract that format description.
        If no specific format is mentioned, return 'default'.

        Text:
        ---
        {file_content}
        ---

        Desired output format:
        """
        
        format_spec = await ResponseFormatter._get_llm_response(prompt)
        
        if format_spec.lower().strip() == 'default':
            return None
            
        return format_spec.strip()

# Example usage and testing
if __name__ == "__main__":
    # Test the formatter
    formatter = ResponseFormatter()
    
    # Test multi-question formatting
    test_query = """
    1. What is the population of India?
    2. What is the population of China?
    3. Which country has higher population?
    """
    
    test_response = """
    1. The population of India is approximately 1.4 billion people.
    2. The population of China is approximately 1.4 billion people.
    3. China has a slightly higher population than India.
    """
    
    formatted = formatter.format_final_answer(test_response, test_query)
    print("Formatted response:")
    print(formatted)
    
    # Test single question
    single_query = "What is the capital of France?"
    single_response = "The capital of France is Paris."
    
    formatted_single = formatter.format_final_answer(single_response, single_query)
    print("\nSingle question formatted:")
    print(formatted_single) 