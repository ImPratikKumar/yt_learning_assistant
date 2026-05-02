import os
from src.vector_store import subtile_md_to_db
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

subtile_md_to_db(api_key=api_key)
print("Vector DB created and persisted.")