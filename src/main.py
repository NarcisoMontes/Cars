from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from google import genai
import pandas as pd
import os
import re
import json
from typing import Dict, Any, Optional

load_dotenv()

DATABASE_URL = f"postgresql://postgres:{os.getenv('DATABASE_PASSWORD')}@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class EnhancedCarAssistant:
    def __init__(self):
        self.conversation_history = []
        self.cached_data = {}
        self.function_registry = self._build_function_registry()

    def _build_function_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of all available database functions"""
        return {
            # Basic Data Retrieval
            "get_all_cars": {
                "description": "Get all cars in the database",
                "params": [],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "get_car_by_id": {
                "description": "Get specific car by ID",
                "params": ["car_id:int"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "search_cars_by_make": {
                "description": "Find cars by manufacturer",
                "params": ["car_make:str"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "search_cars_by_model": {
                "description": "Find cars by model name",
                "params": ["car_model:str"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "search_cars_by_year_range": {
                "description": "Find cars within year range",
                "params": ["start_year:int", "end_year:int"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "get_cars_count_by_year": {
                "description": "Count cars grouped by year",
                "params": [],
                "columns": ['year', 'car_count']
            },
            
            # Aggregation Functions
            "get_cars_count_by_make": {
                "description": "Count cars grouped by manufacturer",
                "params": [],
                "columns": ['make', 'car_count']
            },
            "get_cars_count_by_cylinder": {
                "description": "Count cars grouped by cylinder count",
                "params": [],
                "columns": ['cylinders', 'car_count']
            },
            "get_avg_mpg_by_cylinder": {
                "description": "Average MPG by cylinder count",
                "params": [],
                "columns": ['cylinders', 'avg_mpg']
            },
            "get_avg_price_by_make": {
                "description": "Average price by manufacturer",
                "params": [],
                "columns": ['make', 'avg_price']
            },
            "get_price_statistics": {
                "description": "Price statistics (min, max, avg, median)",
                "params": [],
                "columns": ['min_price', 'max_price', 'avg_price', 'median_price']
            },
            
            # Filtering Functions
            "get_cars_by_price_range": {
                "description": "Filter cars by price range",
                "params": ["min_price:float", "max_price:float"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "get_cars_by_mpg_range": {
                "description": "Filter cars by MPG range",
                "params": ["min_mpg:float", "max_mpg:float"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "get_cars_by_horsepower_range": {
                "description": "Filter cars by horsepower range",
                "params": ["min_hp:int", "max_hp:int"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "get_electric_cars": {
                "description": "Get all electric cars (0 cylinders)",
                "params": [],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "get_most_fuel_efficient": {
                "description": "Get most fuel efficient cars",
                "params": ["limit_count:int=10"],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            
            # Comparison Functions
            "compare_makes": {
                "description": "Compare two manufacturers",
                "params": ["make1:str", "make2:str"],
                "columns": ['make', 'avg_price', 'avg_mpg', 'avg_horsepower', 'car_count']
            },
            "get_best_value_cars": {
                "description": "Get best value cars (performance/price ratio)",
                "params": [],
                "columns": ['id', 'make', 'model', 'value_score', 'price', 'mpg']
            },
            "get_performance_leaders": {
                "description": "Get highest horsepower cars",
                "params": [],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            "get_economy_leaders": {
                "description": "Get most fuel efficient cars",
                "params": [],
                "columns": ['id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price']
            },
            
            # Advanced Analytics
            "get_correlations": {
                "description": "Get correlations between metrics",
                "params": [],
                "columns": ['metric', 'correlation_with_price', 'correlation_with_mpg']
            },
            "get_market_segments": {
                "description": "Get market segments by price range",
                "params": [],
                "columns": ['segment', 'price_range', 'car_count', 'avg_mpg', 'avg_horsepower']
            },
            "get_cars_by_cylinder_detail": {
                "description": "Get detailed info for cars with specific cylinder count",
                "params": ["cyl:int"],
                "columns": ['make', 'model', 'mpg', 'price']
            }
        }

    def _parse_function_call(self, func_call_str: str) -> tuple:
        """Parse function call string to extract name and parameters"""
        # Remove 'CALL:' prefix if present
        func_call_str = re.sub(r'^CALL:\s*', '', func_call_str.strip())
        
        # Extract function name and parameters
        match = re.match(r'(\w+)\((.*?)\)', func_call_str)
        if not match:
            return None, []
        
        func_name = match.group(1)
        params_str = match.group(2).strip()
        
        # Parse parameters
        params = []
        if params_str:
            # Simple parameter parsing - handles strings, numbers
            for param in params_str.split(','):
                param = param.strip()
                if param.startswith('"') and param.endswith('"'):
                    params.append(param[1:-1])  # String parameter
                elif param.startswith("'") and param.endswith("'"):
                    params.append(param[1:-1])  # String parameter
                elif param.isdigit():
                    params.append(int(param))  # Integer parameter
                elif '.' in param and param.replace('.', '').isdigit():
                    params.append(float(param))  # Float parameter
                else:
                    params.append(param)  # Raw parameter
        
        return func_name, params

    def _validate_and_convert_params(self, func_name: str, params: list) -> list:
        """Validate and convert parameters based on function signature"""
        if func_name not in self.function_registry:
            raise ValueError(f"Unknown function: {func_name}")
        
        expected_params = self.function_registry[func_name]["params"]
        converted_params = []
        
        for i, param in enumerate(params):
            if i >= len(expected_params):
                break  # Ignore extra parameters
            
            param_def = expected_params[i]
            param_name, param_type = param_def.split(':')
            
            # Handle optional parameters
            if '=' in param_type:
                param_type, default_val = param_type.split('=')
            
            # Convert parameter to correct type
            if param_type == 'int':
                converted_params.append(int(param))
            elif param_type == 'float':
                converted_params.append(float(param))
            elif param_type == 'str':
                converted_params.append(str(param))
            else:
                converted_params.append(param)
        
        return converted_params

    def execute_function_call(self, func_call_str: str) -> pd.DataFrame:
        """Execute database function from LLM output"""
        try:
            func_name, params = self._parse_function_call(func_call_str)
            if not func_name:
                return f"Error: Could not parse function call: {func_call_str}"
            
            if func_name not in self.function_registry:
                return f"Error: Unknown function: {func_name}"
            
            # Validate and convert parameters
            converted_params = self._validate_and_convert_params(func_name, params)
            columns = self.function_registry[func_name]["columns"]
            
            # Build SQL call
            if converted_params:
                param_placeholders = ', '.join([f':param{i}' for i in range(len(converted_params))])
                sql = f"SELECT * FROM {func_name}({param_placeholders})"
                param_dict = {f'param{i}': param for i, param in enumerate(converted_params)}
            else:
                sql = f"SELECT * FROM {func_name}()"
                param_dict = {}
            
            # Execute query
            with engine.connect() as conn:
                result = conn.execute(text(sql), param_dict)
                return pd.DataFrame(result.fetchall(), columns=columns)
                
        except Exception as e:
            return f"Error executing {func_call_str}: {str(e)}"

    def _format_currency(self, value) -> str:
        """Format currency values"""
        return f"${value:,.2f}" if pd.notnull(value) else "N/A"

    def _format_data_for_display(self, df: pd.DataFrame, func_name: str) -> str:
        """Format DataFrame for conversational display"""
        if isinstance(df, str):  # Error message
            return df
        
        if df.empty:
            return "No results found."
        
        # Format based on function type
        if 'price' in df.columns:
            df = df.copy()
            df['price'] = df['price'].apply(self._format_currency)
        
        if 'avg_price' in df.columns:
            df = df.copy()
            df['avg_price'] = df['avg_price'].apply(self._format_currency)
        
        if 'min_price' in df.columns or 'max_price' in df.columns:
            df = df.copy()
            for col in ['min_price', 'max_price', 'avg_price', 'median_price']:
                if col in df.columns:
                    df[col] = df[col].apply(self._format_currency)
        
        # Convert DataFrame to readable string
        if len(df) <= 10:
            return df.to_string(index=False)
        else:
            return f"Found {len(df)} results. Here are the first 10:\n{df.head(10).to_string(index=False)}"

    def _build_function_descriptions(self) -> str:
        """Build formatted string of all available functions"""
        descriptions = []
        for func_name, info in self.function_registry.items():
            params_str = f"({', '.join(info['params'])})" if info['params'] else "()"
            descriptions.append(f"- {func_name}{params_str}: {info['description']}")
        return '\n'.join(descriptions)

    def chat(self, user_query: str):
        print(f"\nUser: {user_query}")

        # Build conversation history
        history = "\n".join([f"Q: {h['user']}\nA: {h['assistant']}" for h in self.conversation_history[-3:]])
        
        # Build cached data summary
        cached_summary = "Recent data available: " + ", ".join(self.cached_data.keys()) if self.cached_data else "No cached data"
        
        # Build comprehensive prompt
        prompt = f"""You are a car database assistant with access to these functions:

{self._build_function_descriptions()}

Conversation History:
{history}

{cached_summary}

User Query: {user_query}

Instructions:
1. If you need data, output exactly: CALL: function_name(param1, param2, ...)
2. You can make multiple function calls if needed
3. Use cached data when appropriate
4. Provide conversational responses using the data
5. For string parameters, use quotes: search_cars_by_make("Toyota")
6. For numeric parameters, no quotes: get_car_by_id(5)

Respond naturally and conversationally."""

        # Get LLM response
        response = client.models.generate_content(
            #model="gemma-3n-e4b-it",
            model="gemma-3-27b-it",
            contents=prompt
        )

        response_text = response.text

        # Check for function calls
        func_calls = re.findall(r'CALL:\s*([^)\n]+\([^)]*\))', response_text)
        
        if func_calls:
            print(f"Assistant: Getting data...")
            
            # Execute all function calls
            all_results = {}
            for func_call in func_calls:
                cache_key = func_call.strip()
                
                if cache_key in self.cached_data:
                    result = self.cached_data[cache_key]
                    print(f"Using cached data for {func_call}")
                else:
                    result = self.execute_function_call(func_call)
                    self.cached_data[cache_key] = result
                    print(f"Executed: {func_call}")
                
                all_results[func_call] = result

            # Generate final response with all data
            results_summary = ""
            for func_call, result in all_results.items():
                func_name = func_call.split('(')[0].replace('CALL:', '').strip()
                formatted_result = self._format_data_for_display(result, func_name)
                results_summary += f"\n\n{func_call} results:\n{formatted_result}"

            final_prompt = f"""User asked: {user_query}

Data retrieved: {results_summary}

Provide a conversational, helpful response using this data. Be specific and include relevant numbers/details."""

            final_response = client.models.generate_content(
                #model="gemma-3n-e4b-it",
                model="gemma-3-27b-it",
                contents=final_prompt
            )
            response_text = final_response.text

        print(f"Assistant: {response_text}")
        
        # Update conversation history
        self.conversation_history.append({
            'user': user_query, 
            'assistant': response_text
        })

    def clear_cache(self):
        """Clear cached data"""
        self.cached_data.clear()
        print("Cache cleared.")

    def show_functions(self):
        """Display all available functions"""
        print("\nAvailable Functions:")
        print(self._build_function_descriptions())

    def start_conversation(self):
        print("🚗 Enhanced Car Assistant - Commands: 'functions', 'clear', 'quit'\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Assistant: Goodbye! 🚗")
                break
            elif user_input.lower() == 'functions':
                self.show_functions()
            elif user_input.lower() == 'clear':
                self.clear_cache()
            else:
                self.chat(user_input)

if __name__ == "__main__":
    assistant = EnhancedCarAssistant()
    assistant.start_conversation()