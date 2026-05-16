"""
LangGraph Graph
---------------
Wires all nodes into a directed graph with conditional routing.

Graph:
  START
    └─► classify_intent
          ├─[clarify]──► extract_facts ──► generate_clarification ──► END
          ├─[recommend]─► extract_facts ──► retrieve ──► rerank ──► gen_reply ──► END
          ├─[refine]────► extract_facts ──► retrieve ──► rerank ──► gen_reply ──► END
          ├─[compare]───► extract_facts ──► compare_assessments ──► END
          ├─[off_topic]─► refuse ──► END
          └─[greeting]──► greet ──► END
"""

import logging
import os
from functools import partial

from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq

from agent.state import (
    AgentState,
    INTENT_CLARIFY, INTENT_RECOMMEND, INTENT_REFINE,
    INTENT_COMPARE, INTENT_OFF_TOPIC, INTENT_GREETING,
)
from agent.nodes import (
    classify_intent,
    extract_facts,
    generate_clarification,
    retrieve_candidates,
    rerank_and_select,
    generate_recommendation_reply,
    compare_assessments,
    refuse,
    greet,
)
from catalog.vector_store import SHLVectorStore

logger = logging.getLogger(__name__)


def build_graph(vector_store: SHLVectorStore) -> any:
    """
    Build and compile the LangGraph state machine.
    Returns a compiled graph ready to invoke.
    """
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    llm = ChatGroq(
        model=groq_model,
        api_key=groq_api_key,
        temperature=0.1,   # Low temp for consistency
        max_tokens=300,
    )

    # ------------------------------------------------------------------
    # Bind dependencies via partial (keeps nodes pure functions in tests)
    # ------------------------------------------------------------------
    _classify = partial(classify_intent, llm=llm)
    _extract = partial(extract_facts, llm=llm)
    _clarify = partial(generate_clarification, llm=llm)
    _retrieve = partial(retrieve_candidates, vector_store=vector_store)
    _rerank = partial(rerank_and_select, llm=llm)
    _gen_reply = partial(generate_recommendation_reply, llm=llm)
    _compare = partial(compare_assessments, llm=llm, vector_store=vector_store)
    _refuse = partial(refuse, llm=llm)
    _greet = partial(greet, llm=llm)

    # ------------------------------------------------------------------
    # Graph definition
    # ------------------------------------------------------------------
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", _classify)
    graph.add_node("extract_facts", _extract)
    graph.add_node("generate_clarification", _clarify)
    graph.add_node("retrieve_candidates", _retrieve)
    graph.add_node("rerank_and_select", _rerank)
    graph.add_node("generate_reply", _gen_reply)
    graph.add_node("compare_assessments", _compare)
    graph.add_node("refuse", _refuse)
    graph.add_node("greet", _greet)

    # Entry point
    graph.add_edge(START, "classify_intent")

    # Conditional routing from classify_intent
    def route_intent(state: AgentState) -> str:
        intent = state.get("intent", INTENT_CLARIFY)
        if intent == INTENT_OFF_TOPIC:
            return "refuse"
        elif intent == INTENT_GREETING:
            return "greet"
        elif intent == INTENT_COMPARE:
            return "extract_facts_for_compare"
        elif intent in (INTENT_RECOMMEND, INTENT_REFINE):
            return "extract_facts"
        else:  # INTENT_CLARIFY
            return "extract_facts_for_clarify"

    # We need separate paths for clarify vs recommend after extract_facts
    # Use two separate extract_facts nodes for routing clarity

    graph.add_node("extract_facts_for_clarify", _extract)
    graph.add_node("extract_facts_for_compare", _extract)

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "extract_facts": "extract_facts",
            "extract_facts_for_clarify": "extract_facts_for_clarify",
            "extract_facts_for_compare": "extract_facts_for_compare",
            "refuse": "refuse",
            "greet": "greet",
        },
    )

    # After extracting facts for clarify: go to clarification
    graph.add_edge("extract_facts_for_clarify", "generate_clarification")

    # After clarification: check if we should switch to recommend
    def route_after_clarify(state: AgentState) -> str:
        # If clarification node switched intent to recommend
        if state.get("intent") == INTENT_RECOMMEND:
            return "retrieve_candidates"
        return END

    graph.add_conditional_edges(
        "generate_clarification",
        route_after_clarify,
        {
            "retrieve_candidates": "retrieve_candidates",
            END: END,
        },
    )

    # After extracting facts for recommend/refine: retrieve
    graph.add_edge("extract_facts", "retrieve_candidates")
    graph.add_edge("retrieve_candidates", "rerank_and_select")
    graph.add_edge("rerank_and_select", "generate_reply")
    graph.add_edge("generate_reply", END)

    # After extracting facts for compare: go to compare
    graph.add_edge("extract_facts_for_compare", "compare_assessments")
    graph.add_edge("compare_assessments", END)

    # Terminal nodes
    graph.add_edge("refuse", END)
    graph.add_edge("greet", END)

    compiled = graph.compile()
    logger.info("LangGraph agent compiled successfully.")
    return compiled


def build_initial_state(messages: list[dict]) -> AgentState:
    """Build the initial state for a conversation turn."""
    return AgentState(
        messages=messages,
        intent=None,
        facts={},
        candidates=[],
        shortlist=[],
        clarifying_question=None,
        reply="",
        end_of_conversation=False,
        turn_count=len(messages),
        agent_turn_count=sum(1 for m in messages if m["role"] == "assistant"),
    )
