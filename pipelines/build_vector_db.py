import os
from src.vector_store import subtile_md_to_db, create_chunk, create_vectordb
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# subtile_md_to_db(api_key=api_key)

chunks = create_chunk()
print("Chunks created...")
create_vectordb(api_key=api_key, chunks=chunks)

print("Vector DB created and persisted.")