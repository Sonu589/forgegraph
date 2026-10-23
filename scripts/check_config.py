#!/usr/bin/env python3
"""Non-secret environment preflight; never prints DATABASE_URL or model credentials."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
mode = os.getenv("APP_MODE", "demo").lower()
print(f"APP_MODE={mode}")
if mode == "demo":
    print("Demo mode: no database is required; retrieval is lexical and fallback analysis is rule-based.")
elif mode == "alloydb":
    print("DATABASE_URL configured:", bool(os.getenv("DATABASE_URL")))
    print("EMBEDDING_MODEL_ID:", os.getenv("EMBEDDING_MODEL_ID", "embeddinggemma"))
    print("GEMMA_API_URL configured:", bool(os.getenv("GEMMA_API_URL")))
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("ERROR: set DATABASE_URL")
else:
    raise SystemExit("APP_MODE must be 'demo' or 'alloydb'")
