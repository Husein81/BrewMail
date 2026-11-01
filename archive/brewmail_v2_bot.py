import os
import time
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.1-flash-lite"

# -----------------------------------------------------------------------------
# Fake "backend" — in a real system these would hit a database or API.
# -----------------------------------------------------------------------------

AUTHENTICATED_USER_EMAIL = "husseinnasrallah2002@gmail.com"

FAKE_ORDERS = {
    "12345": {
        "status": "shipped", 
        "tracking": "DHL-ABC123", 
        "eta": "2026-10-18",
        "owner":AUTHENTICATED_USER_EMAIL
    },
    "67890": {
        "status": "processing", 
        "tracking": None,         
        "eta": "2026-10-22","owner":"ceo@brewmail.com"
    },
    "11111": {
        "status": "delivered", 
        "tracking": "DHL-ZZZ999",
        "eta": "2026-10-10", 
        "owner":AUTHENTICATED_USER_EMAIL
    },
}

def get_order_status(order_id: str) -> dict:
    owner = AUTHENTICATED_USER_EMAIL 
    print(f"  [TOOL CALL] get_order_status(order_id={order_id!r})")
    if order_id in FAKE_ORDERS and owner == FAKE_ORDERS.get(order_id).get("owner"):
        return {"order_id": order_id, **FAKE_ORDERS[order_id]}
    return {"error": "order_not_found or not authorized", "order_id": order_id}

# This comes from your auth system — login session, JWT, OAuth, etc.
# NOT from the LLM. NOT from the user message. From trusted infrastructure.

SUBSCRIPTIONS = {
    "husseinnasrallah2002@gmail.com": {"status": "active", "owner": "husseinnasrallah2002@gmail.com"},
    "ceo@brewmail.com": {"status": "active", "owner": "ceo@brewmail.com"},
}

def cancel_subscription() -> dict:
    email = AUTHENTICATED_USER_EMAIL

    print(f"  [TOOL CALL] cancel_subscription(email={email!r})")
    
    # Authorization: can the authenticated user cancel this subscription?
    sub = SUBSCRIPTIONS.get(email)
    if sub is None:
        return {"error": "subscription_not_found", "email": email}
    if sub["owner"] != AUTHENTICATED_USER_EMAIL:
        return {
            "error": "unauthorized",
            "message": f"The authenticated user is not permitted to cancel {email}."
        }
    
    sub["status"] = "cancelled"
    return {"success": True, "email": email, "refund_pending": True}

# Map tool names to the actual Python functions.
TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "cancel_subscription": cancel_subscription,
}

# -----------------------------------------------------------------------------
# Tool declarations (what the model sees)
# -----------------------------------------------------------------------------

tools = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_order_status",
        description="Look up the current status of a BrewMail order by its order ID. Returns status, tracking number, and estimated delivery date.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "order_id": types.Schema(
                    type="STRING",
                    description="The BrewMail order ID, usually 5 digits (e.g. '12345').",
                ),
            },
            required=["order_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="cancel_subscription",
        description="Cancel the BrewMail subscription for a given email address. Returns success status and whether a refund is pending.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
            },
        ),
    ),
])]

# -----------------------------------------------------------------------------
# System prompt — now the bot CAN do things, so the "I can't" rules change.
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a customer support agent for BrewMail, a coffee subscription service.

You have access to the following tools:
- get_order_status(order_id): look up a BrewMail order.
- cancel_subscription(email): cancel a subscription for an email address.

RULES:
- Only call a tool when the user has clearly asked for the action it performs.
- When the user asks about an order status, call get_order_status with the order ID they provide.
- When the user asks to cancel their subscription, FIRST confirm their email address with them before calling cancel_subscription.
- Never fabricate order IDs, emails, statuses, or tracking numbers. If information is missing, ask the user.
- If a tool returns an error (e.g. order_not_found), tell the user clearly rather than inventing a result.
"""

# -----------------------------------------------------------------------------
# Retry wrapper (same as before)
# -----------------------------------------------------------------------------

def call_llm(contents):
    for attempt in range(5):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=tools,
                    max_output_tokens=2048,
                ),
            )
        except errors.ServerError:
            time.sleep(2 ** attempt)
        except errors.ClientError as e:
            if e.code == 429:
                time.sleep(2 ** attempt * 2)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

# -----------------------------------------------------------------------------
# The agent loop
# -----------------------------------------------------------------------------

def run_turn(contents, user_input):
    """Add the user's message, run the tool loop until the model gives text."""
    contents.append({"role": "user", "parts": [{"text": user_input}]})

    while True:
        response = call_llm(contents)
        candidate_content = response.candidates[0].content
        part = candidate_content.parts[0]

        # Case 1: model wants to call a tool
        if hasattr(part, "function_call") and part.function_call is not None and part.function_call.name:
            fc = part.function_call

            # ✅ Append the WHOLE Content object returned by the model.
            # This preserves thought_signature and any other metadata Gemini needs.
            contents.append(candidate_content)

            # Run the tool
            fn = TOOL_FUNCTIONS.get(fc.name)
            if fn is None:
                result = {"error": "unknown_tool", "name": fc.name}
            else:
                try:
                    result = fn(**dict(fc.args))
                except Exception as e:
                    result = {"error": "tool_execution_failed", "message": str(e)}

            # Feed the result back as a function_response
            contents.append({
                "role": "user",
                "parts": [{"function_response": {"name": fc.name, "response": result}}],
            })
            continue

        # Case 2: model gave a text answer — done
        text = response.text
        contents.append(candidate_content)   # same principle: append the whole thing
        return text

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

contents = []
print("BrewMail support v2 — type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    try:
        reply = run_turn(contents, user_input)
    except Exception as e:
        print(f"[ERROR: {e}]")
        continue
    print(f"\nAssistant: {reply}\n")