from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from google import genai
import pandas as pd
import os
import json
import re

load_dotenv()

# Database connection
DATABASE_URL = f"postgresql://postgres:{os.getenv('DATABASE_PASSWORD')}@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)

# Gemma client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_cars_by_cylinder():
    """Get car count by cylinder using PostgreSQL function"""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM get_cars_by_cylinder()"))
        return pd.DataFrame(result.fetchall(), columns=['cylinders', 'car_count'])

def get_avg_mpg_by_cylinder():
    """Get average MPG by cylinder count using PostgreSQL function"""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM get_avg_mpg_by_cylinder()"))
        return pd.DataFrame(result.fetchall(), columns=['cylinders', 'avg_mpg'])

def parse_function_call(response_text):
    """Extract function call from Gemma response"""
    # Look for function call pattern
    pattern = r'FUNCTION_CALL:\s*(\w+)'
    match = re.search(pattern, response_text)
    return match.group(1) if match else None

def execute_function(function_name):
    """Execute the requested function"""
    if function_name == "get_cars_by_cylinder":
        return get_cars_by_cylinder()
    elif function_name == "get_avg_mpg_by_cylinder":
        return get_avg_mpg_by_cylinder()
    else:
        raise ValueError(f"Unknown function: {function_name}")

def chat_with_gemma(user_query):
    """Handle user query with Gemma using prompt engineering"""
    
    system_prompt = """You are a database assistant. You have access to two functions:

- get_cars_by_cylinder(): Returns count of cars grouped by cylinder count
- get_avg_mpg_by_cylinder(): Returns average MPG grouped by cylinder count

ONLY call these functions when users ask about cars by cylinders or cylinder-related data.

For car counts by cylinders, respond with exactly:
FUNCTION_CALL: get_cars_by_cylinder

For average MPG by cylinders, respond with exactly:
FUNCTION_CALL: get_avg_mpg_by_cylinder

For any other query (weather, general questions, etc.), just explain you can only help with car cylinder data. Do NOT call any function."""

    full_prompt = f"{system_prompt}\n\nUser: {user_query}\nAssistant:"
    
    response = client.models.generate_content(
        model="gemma-3n-e4b-it",
        contents=full_prompt
    )
    
    response_text = response.text
    print(f"Gemma response: {response_text}")
    
    # Check if Gemma wants to call the function
    function_name = parse_function_call(response_text)
    
    if function_name in ["get_cars_by_cylinder", "get_avg_mpg_by_cylinder"]:
        print(f"Executing: {function_name}")
        result = execute_function(function_name)
        print("Function result:")
        print(result)
        return result
    else:
        return response_text

# Test the system
if __name__ == "__main__":
    queries = [
        "Show me cars by cylinder count",
        "What's the average MPG by cylinder?",
        "How many cars do we have by cylinders?",
        "What's the fuel efficiency by cylinder count?",
        "What's the weather like?"  # Should not trigger function
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        chat_with_gemma(query)
        print()