# app/multi_chatbot_platform.py
import streamlit as st
import os, json, re
from dotenv import load_dotenv
load_dotenv()

from connectors import pinecone_client
from embedding import embed_texts
from connectors.planner.planner import plan_steps_for_query
from connectors.planner.executor import execute_plan
from utils import best_sentences_for_query

RELEVANCE_THRESHOLD = 0.10

DEFAULT_BOTS = {
    "customer_service": {
        "display_name": "Customer Service",
        "db_type": "Pinecone",
        "index_name": os.getenv("PINECONE_INDEX_CUSTOMER_SERVICE", "multi-chatbot-dense"),
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "persona": "Empathetic, helpful, patient tone.",
        "prompt_template": "You are a customer support assistant. Use only retrieved documents to answer clearly and politely."
    },

    "ecommerce": {
        "display_name": "E-commerce",
        "db_type": "Pinecone",
        "index_name": os.getenv("PINECONE_INDEX_ECOMMERCE", "ecommerce"),
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "persona": "Sales-oriented, friendly, highlight deals and recommendations.",
        "prompt_template": "You are a shopping assistant. Use only product documents and pricing information."
    },

    "saas": {
        "display_name": "SaaS Platforms",
        "db_type": "Pinecone",
        "index_name": os.getenv("PINECONE_INDEX_SAAS", "saas"),
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "persona": "Technical, step-by-step, precise.",
        "prompt_template": "You are a SaaS support engineer. Use runbooks and troubleshooting documents only."
    },

    "internal": {
        "display_name": "Internal Teams",
        "db_type": "Pinecone",
        "index_name": os.getenv("PINECONE_INDEX_INTERNAL", "internal"),
        "model": "microsoft/Phi-3-medium-4k-instruct",
        "persona": "Formal, concise, security-conscious.",
        "prompt_template": "You assist internal teams. Use only internal memos and policies."
    }
}


def init_state():
    if "bots" not in st.session_state:
        st.session_state.bots = {}
        for k,v in DEFAULT_BOTS.items():
            st.session_state.bots[k] = {
                **v,
                "retrieved_documents": [],
                "memory_context": {},
                "session_history": []
            }
    if "active_bot" not in st.session_state:
        st.session_state.active_bot = "customer_service"

init_state()
st.set_page_config(layout="wide", page_title="Multi-Chatbot Platform")
st.title("Multi-Chatbot Platform — Pinecone Backend")

# Sidebar: choose bot only
with st.sidebar:
    st.header("Chatbots")
    bot_choice = st.radio("Select active chatbot", list(st.session_state.bots.keys()), format_func=lambda k: st.session_state.bots[k]["display_name"])
    bot = st.session_state.bots[bot_choice]
    
    st.markdown("---")
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload JSON data", type=['json'])
    if uploaded_file and st.button("Load to Database"):
        try:
            data = json.load(uploaded_file)
            index_name = bot["index_name"]
            
            success_count = 0
            for item in data:
                try:
                    embedding = embed_texts([item.get("text", "")])[0]
                    pinecone_client.pinecone_upsert_to_index(
                        index_name,
                        item.get("id"), 
                        embedding, 
                        {"title": item.get("title", ""), "text": item.get("text", "")}
                    )
                    success_count += 1
                except Exception as e:
                    st.error(f"Failed to load {item.get('id')}: {e}")
            
            st.success(f"✓ Loaded {success_count}/{len(data)} documents to Pinecone index '{index_name}'")
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Main layout


st.header("Ask the bot")
query = st.text_area("Your query", height=140, key="user_query")

if st.button("Send query", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        bot["session_history"].append({"role":"user","content":query})
        
        with st.spinner("Planning steps..."):
            steps = plan_steps_for_query(query)
            st.info("**Plan:**")
            for s in steps:
                st.write("- " + s)

        # Generate embedding for query
        with st.spinner("Generating query embedding..."):
            try:
                emb_vec = embed_texts([query])[0]
            except Exception as e:
                st.error(f"Embedding error: {e}")
                emb_vec = None

        # Retrieve from Pinecone
        retrieved = []
        if emb_vec:
            with st.spinner(f"Searching Pinecone index '{bot['index_name']}'..."):
                retrieved = pinecone_client.pinecone_query_index(
                    bot["index_name"],
                    vectors=emb_vec, 
                    top_k=4
                )
                st.write(f"Found {len(retrieved)} documents")

        # Check relevance
        if not retrieved or retrieved[0][0] < RELEVANCE_THRESHOLD:
            clar = ("I couldn't find sufficiently relevant documents in this bot's database. "
                    "Per system rules I cannot guess. Please clarify or provide supporting documents.")
            st.warning(clar)
            bot["session_history"].append({"role":"assistant","content":clar})
        else:
            with st.spinner("Generating grounded answer..."):
                resp = execute_plan(steps, retrieved, query)
                
                st.success("**Answer (grounded):**")
                st.write(resp["answer"])
                
                st.markdown("**Citations:**")
                for c in resp["citations"]:
                    st.write("- " + c)
                
                bot["session_history"].append({
                    "role":"assistant",
                    "content":resp["answer"], 
                    "citations": resp["citations"]
                })

st.markdown("---")
if st.button("Show debug info"):
    st.json({
        "index_name": bot["index_name"],
        "memory_context": bot["memory_context"], 
        "session_history": bot["session_history"][-10:]
    })

