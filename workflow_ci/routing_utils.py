"""
Shared routing constants dan helper functions.
Di-import oleh modelling.py, modelling_tuning.py, modelling_sbert.py.
Ubah threshold DI SINI — otomatis berlaku di semua script.
"""

import pandas as pd

HIGH_THR = 0.55
MID_THR  = 0.30

COST_MAP = {
    "template_handler": 0.001,
    "rag_pipeline":     0.010,
    "llm_escalation":   0.050,
}


def route_decision(confidence: float) -> str:
    if confidence >= HIGH_THR:
        return "template_handler"
    elif confidence >= MID_THR:
        return "rag_pipeline"
    return "llm_escalation"


def estimate_cost_savings(routing_decisions: pd.Series) -> dict:
    actual_cost   = routing_decisions.map(COST_MAP).sum()
    baseline_cost = len(routing_decisions) * COST_MAP["llm_escalation"]
    savings_pct   = (1 - actual_cost / baseline_cost) * 100
    return {
        "actual_cost_usd":   round(actual_cost, 4),
        "baseline_cost_usd": round(baseline_cost, 4),
        "savings_percent":   round(savings_pct, 2),
        "template_pct":      round((routing_decisions == "template_handler").mean() * 100, 2),
        "rag_pct":           round((routing_decisions == "rag_pipeline").mean() * 100, 2),
        "llm_pct":           round((routing_decisions == "llm_escalation").mean() * 100, 2),
    }
