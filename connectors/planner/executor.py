# app/planner/executor.py
from typing import List, Tuple, Dict
from utils import best_sentences_for_query
from google.generativeai import GenerativeModel
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
def execute_plan(plan: List[str], retrieved_docs: List[Tuple[float, Dict]], query: str) -> Dict:
    # Use top docs (score descending) to build a grounded answer
    used = [ (s,d) for s,d in retrieved_docs if s is not None ]
    used = sorted(used, key=lambda x: x[0], reverse=True)[:3]
    parts = []
    citations = []
    for score, doc in used:
        parts.extend(best_sentences_for_query(doc.get("text",""), query, k=2))
        citations.append(doc.get("title","(untitled)"))
    answer = " ".join(parts).strip() or "I couldn't extract a concise answer from the documents."
    #use gemini llm to refine answer based on plan
    llm_prompt = f"""You are an expert assistant. Based on the following plan and extracted information, provide a concise and accurate answer to the user's query.
Plan:
{', '.join(plan)}
Extracted Information:
{" ".join(parts)}
User Query:
{query}
Provide the final answer below:
Final Answer:
"""
    # Here you would call the Gemini LLM API with llm_prompt to get the refined answer

    model = GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(llm_prompt)
    refined_answer = response.text
    return {"answer": refined_answer, "citations": citations}






