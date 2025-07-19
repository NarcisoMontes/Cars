from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from google import genai
import pandas as pd
import os
import re

load_dotenv()

DATABASE_URL = f"postgresql://postgres:{os.getenv('DATABASE_PASSWORD')}@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class SimpleCarAssistant:
    def __init__(self):
        self.conversation_history = []
        self.cached_data = {}

    # Database functions
    def get_cars_by_cylinder(self):
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM get_cars_by_cylinder()"))
            return pd.DataFrame(result.fetchall(), columns=['cylinders', 'car_count'])

    def get_avg_mpg_by_cylinder(self):
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM get_avg_mpg_by_cylinder()"))
            return pd.DataFrame(result.fetchall(), columns=['cylinders', 'avg_mpg'])

    def get_cars_by_cylinder_detail(self, cylinders):
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM get_cars_by_cylinder_detail(:cyl)"), {"cyl": cylinders})
            return pd.DataFrame(result.fetchall(), columns=['make', 'model', 'mpg', 'price'])

    def execute_function_call(self, func_call):
        """Execute function from LLM output"""
        try:
            if func_call.startswith("get_cars_by_cylinder_detail("):
                cyl = int(re.search(r'\((\d+)\)', func_call).group(1))
                return self.get_cars_by_cylinder_detail(cyl)
            elif func_call == "get_cars_by_cylinder()":
                return self.get_cars_by_cylinder()
            elif func_call == "get_avg_mpg_by_cylinder()":
                return self.get_avg_mpg_by_cylinder()
        except Exception as e:
            return f"Error: {e}"

    def chat(self, user_query):
        print(f"\nUser: {user_query}")

        # Build context
        history = "\n".join([f"Q: {h['user']}\nA: {h['assistant']}" for h in self.conversation_history[-2:]])
        cached = str(self.cached_data) if self.cached_data else "No cached data"

        prompt = f"""You are a car database assistant. Available functions:
- get_cars_by_cylinder(): Returns car counts by cylinder
- get_avg_mpg_by_cylinder(): Returns average MPG by cylinder
- get_cars_by_cylinder_detail(X): Returns specific cars with X cylinders

History: {history}
Cached data: {cached}

User: {user_query}

If you need data, output exactly: CALL: function_name()
Then provide a conversational response using the data.
If you have the data already, just respond conversationally."""

        response = client.models.generate_content(
            model="gemma-3n-e4b-it",
            contents=prompt
        )

        response_text = response.text

        # Check for function call
        func_match = re.search(r'CALL:\s*([^)\n]+\([^)]*\))', response_text)
        if func_match:
            func_call = func_match.group(1)
            print(f"Assistant: Getting data...")

            result = self.execute_function_call(func_call)
            self.cached_data[func_call] = result

            # Get final response with data
            final_prompt = f"User asked: {user_query}\nData retrieved: {result}\n\nProvide a conversational response:"
            final_response = client.models.generate_content(
                model="gemma-3n-e4b-it",
                contents=final_prompt
            )
            response_text = final_response.text

        print(f"Assistant: {response_text}")
        self.conversation_history.append({'user': user_query, 'assistant': response_text})

    def start_conversation(self):
        print("🚗 Car Assistant - Type 'quit' to exit\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("Assistant: Goodbye!")
                break
            self.chat(user_input)

if __name__ == "__main__":
    assistant = SimpleCarAssistant()
    assistant.start_conversation()