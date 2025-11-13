# app/planner/planner.py
def plan_steps_for_query(query: str):
    steps = []
    steps.append("Retrieve relevant documents from the active chatbot's database.")
    q = query.lower()
    if any(w in q for w in ["error","fail","issue","trouble","bug","exception"]):
        steps.append("Extract troubleshooting steps from the docs.")
        steps.append("Compose step-by-step resolution and suggested checks.")
    elif any(w in q for w in ["price","cost","buy","purchase","coupon","discount","offer"]):
        steps.append("Extract product/pricing docs and confirm current offers.")
    else:
        steps.append("Extract concise facts from the most relevant docs and summarize with citations.")
    return steps
