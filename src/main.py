from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model = "gemma-3n-e4b-it", contents = "Explain how AI works in a few words."
)
print(response.text)