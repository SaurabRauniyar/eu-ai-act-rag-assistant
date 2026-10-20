"""
Test the Groq API connection.

This script only checks whether:
1. The API key can be loaded from .env
2. Python can connect to Groq
3. An LLM can return a response
"""

import os

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

print("Groq API key loaded successfully.")


# ---------------------------------------------------------
# Create Groq client
# ---------------------------------------------------------

client = Groq(api_key=api_key)


# ---------------------------------------------------------
# Send a very simple test question
# ---------------------------------------------------------

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one sentence."
        }
    ]
)


# ---------------------------------------------------------
# Print LLM response
# ---------------------------------------------------------

answer = response.choices[0].message.content

print("\nLLM response:")
print(answer)