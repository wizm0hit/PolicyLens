import os

from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")


if not API_KEY:

    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please add it to your .env file."
    )


# ---------------------------------------------------------
# Gemini Client & Models
# ---------------------------------------------------------

CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.6-flash",
]

def get_client():
    """Create a fresh Gemini client instance to avoid closed-client errors."""
    return genai.Client(api_key=API_KEY)


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

    last_error = None

    for model_name in CANDIDATE_MODELS:
        for attempt in range(2):
            try:
                client = get_client()
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