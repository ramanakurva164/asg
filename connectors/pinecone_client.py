# app/connectors/pinecone_client.py
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
try:
    PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
except:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

try:
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
    print(f"DEBUG: Pinecone client initialized")
except Exception as e:
    print(f"DEBUG: Failed to initialize Pinecone: {e}")
    pc = None

def get_or_create_index(index_name):
    """Get or create a Pinecone index"""
    if pc is None:
        raise Exception("Pinecone not initialized")
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        import time
        time.sleep(10)
    
    return pc.Index(index_name)

def pinecone_upsert_to_index(index_name, doc_id, vector, metadata):
    """Upsert document to specific index"""
    try:
        index = get_or_create_index(index_name)
        index.upsert(vectors=[(doc_id, vector, metadata)])
        print(f"✓ Added {doc_id} to index {index_name}")
    except Exception as e:
        raise Exception(f"Pinecone upsert error: {e}")

def pinecone_query_index(index_name, vectors, top_k=4):
    """Query specific Pinecone index"""
    try:
        index = get_or_create_index(index_name)
        results = index.query(
            vector=vectors,
            top_k=top_k,
            include_metadata=True
        )
        
        retrieved = []
        for match in results.matches:
            retrieved.append((match.score, {
                'id': match.id,
                'title': match.metadata.get('title', ''),
                'text': match.metadata.get('text', '')
            }))
        
        return retrieved
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []
