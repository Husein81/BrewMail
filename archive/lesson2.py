import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain what a REST API is in exactly 2 sentences."
)

print(response)
print("---")
print("Text only:", response.text)