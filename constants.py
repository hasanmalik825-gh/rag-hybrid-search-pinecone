import os

IP_WHITELIST = os.getenv("IP_WHITELIST") or ["127.0.0.1"]
PORT = os.getenv("PORT") or 8000
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
RETRIEVE_K = 3