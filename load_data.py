import json
import os
from dotenv import load_dotenv
load_dotenv()

from connectors import pinecone_client
from embedding import embed_texts

def load_json_to_pinecone_index(json_file_path, index_name):
    """Load JSON data into specific Pinecone index."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Loading {len(data)} documents to Pinecone index '{index_name}'")
    print(f"{'='*60}")
    
    success = 0
    failed = 0
    
    for item in data:
        doc_id = item.get("id")
        text = item.get("text", "")
        title = item.get("title", "")
        
        try:
            embedding = embed_texts([text])[0]
            pinecone_client.pinecone_upsert_to_index(
                index_name,
                doc_id,
                embedding,
                {"title": title, "text": text}
            )
            success += 1
            
        except Exception as e:
            failed += 1
            print(f"✗ Failed {doc_id}: {e}")
    
    print(f"\nSummary: {success} succeeded, {failed} failed\n")

def load_all_default_indexes():
    print("\n🚀 Starting data load to Pinecone indexes...\n")
    
    load_json_to_pinecone_index(
        "data/customer_service.json",
        os.getenv("PINECONE_INDEX_CUSTOMER_SERVICE")
    )
    load_json_to_pinecone_index(
        "data/ecommerce.json",
        os.getenv("PINECONE_INDEX_ECOMMERCE")
    )
    load_json_to_pinecone_index(
        "data/saas.json",
        os.getenv("PINECONE_INDEX_SAAS")
    )
    load_json_to_pinecone_index(
        "data/internal.json",
        os.getenv("PINECONE_INDEX_INTERNAL")
    )
    
    print("\n✅ All data loaded successfully!\n")


if __name__ == "__main__":
    # still works if you run python load_data.py locally
    load_all_default_indexes()
    print("\n✅ All data loaded successfully!\n")
