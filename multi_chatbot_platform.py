import streamlit as st
import os, json, re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from connectors import pinecone_client
from embedding import embed_texts
from connectors.planner.planner import plan_steps_for_query
from connectors.planner.executor import execute_plan
from utils import best_sentences_for_query
from load_data import load_all_default_indexes

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
                "sessions": {},  # Each bot maintains its own sessions
                "active_session_id": None  # Each bot has its own active session
            }
    
    if "active_bot" not in st.session_state:
        st.session_state.active_bot = "customer_service"
@st.cache_resource
def initialize_pinecone_data():
    load_all_default_indexes()
    return "done"

def create_new_session(bot_key):
    """Create a new chat session for a specific bot"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    bot = st.session_state.bots[bot_key]
    bot["sessions"][session_id] = {
        "name": f"Chat {len(bot['sessions']) + 1}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }
    return session_id

def get_active_session(bot_key):
    """Get or create active session for a specific bot"""
    bot = st.session_state.bots[bot_key]
    
    # If no sessions exist for this bot, create one
    if not bot["sessions"]:
        session_id = create_new_session(bot_key)
        bot["active_session_id"] = session_id
    
    # If active_session_id is None or invalid for this bot, use first session
    if (bot["active_session_id"] is None or 
        bot["active_session_id"] not in bot["sessions"]):
        bot["active_session_id"] = list(bot["sessions"].keys())[0]
    
    return bot["active_session_id"]

init_state()
st.set_page_config(layout="wide", page_title="Multi-Chatbot Platform")
initialize_pinecone_data()
st.title("Multi-Chatbot Platform — Pinecone Backend")


# Sidebar
with st.sidebar:
    st.header("🤖 Chatbots")
    
    # Bot selection
    bot_choice = st.radio(
        "Select active chatbot", 
        list(st.session_state.bots.keys()), 
        format_func=lambda k: st.session_state.bots[k]["display_name"],
        key="bot_selector"
    )
    
    # Update active bot
    if st.session_state.active_bot != bot_choice:
        st.session_state.active_bot = bot_choice
    
    bot = st.session_state.bots[bot_choice]
    
    # Show bot info
    with st.expander("ℹ️ Bot Info"):
        st.write(f"**Index:** {bot['index_name']}")
        st.write(f"**Model:** {bot['model']}")
        st.write(f"**Persona:** {bot['persona']}")
    
    st.markdown("---")
    st.subheader(f"💬 {bot['display_name']} Sessions")
    
    # Show total sessions for this bot
    st.caption(f"Total sessions: {len(bot['sessions'])}")
    
    # New session button
    if st.button("➕ New Chat Session", use_container_width=True):
        new_session_id = create_new_session(bot_choice)
        bot["active_session_id"] = new_session_id
        st.rerun()
    
    # Display sessions for current bot
    sessions = bot["sessions"]
    if sessions:
        session_options = {
            sid: f"{sdata['name']} ({sdata['created_at']})" 
            for sid, sdata in sessions.items()
        }
        
        current_active = get_active_session(bot_choice)
        
        selected_session = st.selectbox(
            "Active Session",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=list(session_options.keys()).index(current_active),
            key=f"session_selector_{bot_choice}"
        )
        
        if selected_session != bot["active_session_id"]:
            bot["active_session_id"] = selected_session
            st.rerun()
        
        # Session options
        with st.expander("⚙️ Session Options"):
            new_name = st.text_input(
                "Rename session",
                value=sessions[selected_session]["name"],
                key=f"rename_{bot_choice}_{selected_session}"
            )
            if st.button("Update Name", key=f"update_{bot_choice}_{selected_session}"):
                sessions[selected_session]["name"] = new_name
                st.success("Session renamed!")
                st.rerun()
            
            # Delete session
            if len(sessions) > 1:
                if st.button("🗑️ Delete Session", key=f"delete_{bot_choice}_{selected_session}"):
                    del sessions[selected_session]
                    bot["active_session_id"] = list(sessions.keys())[0]
                    st.success("Session deleted!")
                    st.rerun()
            else:
                st.info("Cannot delete the last session")
    
    st.markdown("---")
    st.subheader("📤 Upload Data")
    uploaded_file = st.file_uploader("Upload JSON data", type=['json'], key=f"upload_{bot_choice}")
    if uploaded_file and st.button("Load to Database", key=f"load_{bot_choice}"):
        try:
            data = json.load(uploaded_file)
            index_name = bot["index_name"]
            
            success_count = 0
            with st.spinner(f"Loading {len(data)} documents..."):
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
            
            st.success(f"✓ Loaded {success_count}/{len(data)} documents to index '{index_name}'")
        except Exception as e:
            st.error(f"Error loading data: {e}")

# Main layout
col_left, col_right = st.columns([1.4, 1])

# Get active session for current bot
active_session_id = get_active_session(bot_choice)
active_session = bot["sessions"][active_session_id]

with col_left:
    
    
    st.markdown("---")
    st.subheader("Chat History")
    
    # Display chat messages
    if not active_session["messages"]:
        st.info(f"No messages in this session yet. Start chatting with {bot['display_name']}!")
    else:
        for msg in active_session["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
                # Show citations for assistant messages
                if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
                    with st.expander("📚 View Citations"):
                        for citation in msg["citations"]:
                            st.write(f"- {citation}")
    
    # Clear session button
    if active_session["messages"]:
        if st.button("🗑️ Clear Chat History", key=f"clear_{bot_choice}"):
            active_session["messages"] = []
            st.rerun()

with col_right:
    st.header("💭 Ask the bot")
    
    # Query input
    query = st.text_area(
        "Your query", 
        height=140, 
        key=f"user_query_{bot_choice}_{active_session_id}",
        placeholder=f"Ask {bot['display_name']} a question..."
    )
    
    if st.button("Send query", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            # Add user message
            active_session["messages"].append({"role": "user", "content": query})
            
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
                active_session["messages"].append({"role": "assistant", "content": clar})
            else:
                with st.spinner("Generating grounded answer..."):
                    resp = execute_plan(steps, retrieved, query)
                    
                    st.success("**Answer (grounded):**")
                    st.write(resp["answer"])
                    
                    st.markdown("**Citations:**")
                    for c in resp["citations"]:
                        st.write("- " + c)
                    
                    active_session["messages"].append({
                        "role": "assistant",
                        "content": resp["answer"], 
                        "citations": resp["citations"]
                    })
            
            st.rerun()

    st.markdown("---")
    
    # Debug info
    # with st.expander("🔍 Debug Info"):
    #     st.json({
    #         "current_bot": bot_choice,
    #         "bot_display_name": bot['display_name'],
    #         "session_id": active_session_id,
    #         "session_name": active_session["name"],
    #         "message_count": len(active_session["messages"]),
    #         "index_name": bot["index_name"],
    #         "total_sessions_for_bot": len(bot["sessions"]),
    #         "all_session_ids": list(bot["sessions"].keys()),
    #         "all_bots": {
    #             k: {
    #                 "total_sessions": len(v["sessions"]),
    #                 "active_session": v.get("active_session_id")
    #             }
    #             for k, v in st.session_state.bots.items()
    #         }
    #     })

