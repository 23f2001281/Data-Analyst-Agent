import pandas as pd
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_pandas_code(df_head: str, user_prompt: str) -> str:
    """
    Uses an LLM to generate Python code for pandas data analysis.
    """
    # Use the more powerful Pro model for code generation
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Construct a clear prompt for the LLM
    system_prompt = f"""
    You are a data analysis expert. Your task is to write a Python script using the pandas library to analyze the given data.
    
    The user will provide the head of a pandas DataFrame to show you its structure, and a prompt describing the analysis they want.

    The DataFrame is already loaded into a variable named `df`.
    
    Your script should not include any code to load the data. It should only contain the pandas operations.
    The final result of your script should be stored in a variable named `result`.
    
    DataFrame Head:
    {df_head}
    
    User's Analysis Prompt:
    {user_prompt}
    
    Write the Python script now.
    """
    
    response = model.generate_content(system_prompt)
    
    code = response.text.strip().replace("```python", "").replace("```", "").strip()
    return code

def run_analysis_task(raw_data: list[dict], user_prompt: str):
    """
    Executes a data analysis task on the given raw data.
    """
    print("--- Starting data analysis task ---")
    
    try:
        # Load the raw data into a pandas DataFrame
        df = pd.DataFrame(raw_data)
        print("DataFrame created successfully:")
        print(df.head())
        
        # Generate the pandas code
        print("\n--- Generating analysis code... ---")
        pandas_code = generate_pandas_code(df.head().to_string(), user_prompt)
        print("Generated Code:\n", pandas_code)
        
        # Execute the generated code
        print("\n--- Executing code... ---")
        local_scope = {'df': df}
        exec(pandas_code, globals(), local_scope)
        result = local_scope.get('result', "No result variable found in executed code.")
        
        print("--- Analysis task completed successfully! ---")
        return result

    except Exception as e:
        print(f"An error occurred during the analysis task: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    print("Analysis Agent module loaded successfully.")
    print("Use run_analysis_task() function to analyze data.")
