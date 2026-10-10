"""Quick test: call Groq with the new model."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from dotenv import load_dotenv
load_dotenv()
from agri_core import call_groq, GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL

print(f"Primary model  : {GROQ_PRIMARY_MODEL}")
print(f"Fallback model : {GROQ_FALLBACK_MODEL}")
print("Testing call_groq()...")

result = call_groq("In one sentence, what is the best kharif crop for black cotton soil in Karnataka?")
print(f"\nResponse: {result[:300]}")
print("\nGroq model test PASSED" if "Groq request failed" not in result else "FAILED")
