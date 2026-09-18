import os

from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


def get_api_key():
    """Retrieve Gemini API key dynamically from environment."""
    return os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------
# Gemini Client & Models
# ---------------------------------------------------------

CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
]

def get_client():
    """Create a fresh Gemini client instance to avoid closed-client errors."""
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------
# Generate grounded answer
# ---------------------------------------------------------

def generate_answer(question, evidence):

    prompt = f"""
You are CareAssist, a hospital policy
knowledge assistant.

Your job is to answer questions ONLY
using the supplied hospital policy evidence.

IMPORTANT RULES:

1. Use ONLY the provided evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent policies, timings, fees,
   procedures, requirements, or exceptions.

4. If the evidence does not contain enough
   information to answer the question,
   say exactly:

"I don't have enough information in the
hospital policy documents to answer that
question."

5. Keep the answer concise and easy to understand.

6. Preserve important steps, requirements,
   timings, and conditions from the evidence.

7. Do not provide medical diagnosis or
   treatment advice.

8. Always provide the source document,
   section, and page when available.


USER QUESTION:

{question}


POLICY EVIDENCE:

{evidence}


ANSWER:
"""

    # -----------------------------------------------------
    # Call Gemini with fallback models & retry
    # -----------------------------------------------------

    if not get_api_key():
        return (
            "GEMINI_API_KEY is not configured on the server. "
            "Please set the GEMINI_API_KEY environment variable in your deployment settings."
        )

    last_error = None

    for model_name in CANDIDATE_MODELS:
        for attempt in range(2):
            try:
                client = get_client()
                if not client:
                    continue
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = e
                print(f"\n[LLM] Error with {model_name} (attempt {attempt + 1}): {e}")
                continue

    print("\nLLM Error (all models exhausted):", last_error)
    return (
        "I'm temporarily unable to generate an answer. "
        "Please try the question again."
    )