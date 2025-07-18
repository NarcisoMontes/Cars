from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from google import genai
import pandas as pd
import os

load_dotenv()

# Database connection
DATABASE_URL = f"postgresql://postgres:{os.getenv('DATABASE_PASSWORD')}@localhost:5432/postgres"
engine = create_engine(DATABASE_URL)

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_cars_by_cylinder():
    """Get car count by cylinder using PostgreSQL function"""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM get_cars_by_cylinder()"))
        return pd.DataFrame(result.fetchall(), columns=['cylinders', 'car_count'])

def get_all_cars():
    """Get all cars from database"""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM cars"))
        return pd.DataFrame(result.fetchall(), columns=[
            'id', 'make', 'model', 'year', 'cylinders', 'horsepower', 'mpg', 'price'
        ])

# Test database connection
if __name__ == "__main__":
    try:
        # Test database
        cars_by_cyl = get_cars_by_cylinder()
        print("Cars by cylinder:")
        print(cars_by_cyl)

        # Test Gemini
        response = client.models.generate_content(
            model="gemma-3n-e4b-it",
            contents="Explain this car data in one sentence."
        )
        print(f"\nGemini response: {response.text}")

    except Exception as e:
        print(f"Error: {e}")
