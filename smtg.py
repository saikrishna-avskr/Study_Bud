from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
            model='gemini-2.0-flash', contents="generate all the resposes in detailed from now onwords. I am using it for learning purpose.",
        )
print(response)