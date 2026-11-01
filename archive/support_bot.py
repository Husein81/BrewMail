import os
import time
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.1-flash-lite"
contents = []
total_tokens = 0

SYSTEM_PROMPT = """
You are a customer support agent for BrewMail, a coffee subscription service.

ROLE AND SCOPE:
- You help with: subscription plans, delivery schedules, pausing/cancelling, and general coffee questions.
- You do NOT help with:  Anything outside the subscription plans, general coffe, delivery schedules, pausing/cancelling. 

BEHAVIOR:
- If asked about anything else, politly decline and redirect to the coffee subscription service topic.

CONSTRAINTS:
- you do NOT have any access to any database right now  so you should never pretende you do. 
- you do NOT have access to user's account
- you do NOT have the ability to do any changes 

if user asks any of the constraints you should responed that you do not have access to them
"""

def generate_with_retry(contents, max_retries=5):
    """Call Gemini with exponential backoff on 5xx / rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                    ,max_output_tokens=1024),
            )
        except errors.ServerError as e:
            # 5xx — transient, retry
            wait = 2 ** attempt   # 1s, 2s, 4s, 8s, 16s
            print(f"[Server error ({e.code}), retrying in {wait}s...]")
            time.sleep(wait)
        except errors.ClientError as e:
            # 4xx — your fault (bad request, bad key, etc.) — do NOT retry
            if e.code == 429:
                # Rate limit — treat like 5xx, wait longer
                wait = 2 ** attempt * 2
                print(f"[Rate limited, retrying in {wait}s...]")
                time.sleep(wait)
            else:
                print(f"[Client error, not retrying: {e}]")
                raise
    raise RuntimeError(f"Gave up after {max_retries} retries")


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    contents.append({"role": "user", "parts": [{"text": user_input}]})

    try:
        response = generate_with_retry(contents)
    except Exception as e:
        print(f"[Request failed: {e}]")
        # Important: pop the user turn so history stays coherent
        contents.pop()
        continue

    reply = response.text
    contents.append({"role": "model", "parts": [{"text": reply}]})

    in_tokens = response.usage_metadata.prompt_token_count
    out_tokens = response.usage_metadata.candidates_token_count
    total_tokens += in_tokens + out_tokens

    print(f"\nAssistant: {reply}")
    print(f"[Tokens: input={in_tokens}, output={out_tokens}, total so far: {total_tokens}]\n")