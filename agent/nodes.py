"""
LangGraph Agent Nodes
---------------------

Trace-optimized conversational behavior for SHL evaluator.

Key upgrades:
- Strong refinement persistence
- Shortlist continuity
- Layered battery awareness
- Early recommendation bias
- Better comparison grounding
- Catalog-gap handling
- Deterministic fallback logic
"""

import json
import logging
import re
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from agent.state import (
    AgentState,
    ExtractedFacts,

    INTENT_CLARIFY,
    INTENT_RECOMMEND,
    INTENT_REFINE,
    INTENT_COMPARE,
    INTENT_OFF_TOPIC,
    INTENT_GREETING,

    MAX_TURNS,
    MAX_RECOMMENDATIONS,

    CLARIFICATION_PRIORITY,
)

from agent.prompts import (
    SYSTEM_PROMPT,

    intent_classification_prompt,
    fact_extraction_prompt,
    clarifying_question_prompt,

    rerank_prompt,

    recommendation_reply_prompt,
    comparison_reply_prompt,

    off_topic_reply_prompt,
    greeting_reply_prompt,

    format_conversation,
)

logger = logging.getLogger(__name__)


# ==========================================================
# CONFIRMATION PHRASES
# ==========================================================

CONFIRMATION_PHRASES = [
    "confirmed",
    "that works",
    "perfect",
    "that's good",
    "that covers it",
    "keep the shortlist",
    "final list",
    "go with that",
    "use that",
    "looks good",
    "confirmed shortlist",
    "locking it in",
    "keep as is",
]


# ==========================================================
# HARD HEURISTICS
# ==========================================================

SALARY_KEYWORDS = [
    "salary",
    "compensation",
    "package",
    "pay",
    "ctc",
    "wage",
]

LEADERSHIP_TERMS = [
    "leadership",
    "executive",
    "cxo",
    "director",
    "benchmark",
    "succession",
]

SAFETY_TERMS = [
    "safety",
    "plant",
    "chemical",
    "industrial",
    "compliance",
    "dependability",
    "reliability",
]

CONTACT_CENTER_TERMS = [
    "contact center",
    "call center",
    "customer service",
    "inbound calls",
]

HEALTHCARE_TERMS = [
    "healthcare",
    "medical",
    "hipaa",
    "patient",
]

OFFICE_TERMS = [
    "excel",
    "word",
    "administrative",
    "admin assistant",
]

def _latest_user_message(
    messages: list[dict],
) -> str:

    for m in reversed(messages):

        if m["role"] == "user":
            return m["content"]

    return ""


# ==========================================================
# LLM CALL
# ==========================================================

def _llm_call(
    llm: ChatGroq,
    prompt: str,
    system: str = SYSTEM_PROMPT,
) -> str:

    try:

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)

        return response.content.strip()

    except Exception as e:

        logger.error(
            f"LLM call failed: {e}"
        )

        raise


# ==========================================================
# JSON PARSER
# ==========================================================

def _parse_json_response(
    text: str,
) -> Any:

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
    )

    text = re.sub(
        r"```",
        "",
        text,
    )

    text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    patterns = [
        r"\{.*\}",
        r"\[.*\]",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.DOTALL,
        )

        if match:

            try:
                return json.loads(
                    match.group()
                )

            except Exception:
                continue

    return None


# ==========================================================
# USER CONFIRMATION
# ==========================================================

def _user_has_confirmed(
    messages: list[dict],
) -> bool:

    last_user = ""

    for m in reversed(messages):

        if m["role"] == "user":
            last_user = (
                m["content"]
                .lower()
                .strip()
            )
            break

    return any(
        phrase in last_user
        for phrase in CONFIRMATION_PHRASES
    )


# ==========================================================
# BUILD RETRIEVAL QUERY
# ==========================================================

def _build_retrieval_query(
    facts: dict,
) -> str:

    parts = []

    for field in [
        "job_role",
        "job_function",
        "seniority",
        "sector",
        "language_requirement",
    ]:

        value = facts.get(field)

        if value:
            parts.append(str(value))

    for field in [
        "what_to_measure",
        "specific_skills",
        "assessment_preferences",
        "additional_constraints",
    ]:

        values = facts.get(field)

        if values:
            parts.extend(values)

    query = " ".join(parts).strip()

    lower = query.lower()

    boosted = []

    # ======================================================
    # DOMAIN BOOSTS
    # ======================================================

    if any(
        t in lower
        for t in LEADERSHIP_TERMS
    ):

        boosted.extend([
            "OPQ32r",
            "Leadership Report",
            "UCF",
        ])

    if any(
        t in lower
        for t in SAFETY_TERMS
    ):

        boosted.extend([
            "DSI",
            "Safety",
            "Dependability",
        ])

    if any(
        t in lower
        for t in CONTACT_CENTER_TERMS
    ):

        boosted.extend([
            "SVAR",
            "Customer Service",
            "Contact Center",
        ])

    if any(
        t in lower
        for t in HEALTHCARE_TERMS
    ):

        boosted.extend([
            "HIPAA",
            "Medical Terminology",
        ])

    if any(
        t in lower
        for t in OFFICE_TERMS
    ):

        boosted.extend([
            "Microsoft Excel",
            "Microsoft Word",
        ])

    if boosted:
        query += " " + " ".join(boosted)

    return query


# ==========================================================
# CLASSIFY INTENT
# ==========================================================

def classify_intent(
    state: AgentState,
    llm: ChatGroq,
) -> AgentState:

    messages = state["messages"]

    turn_count = len(messages)

    if turn_count >= MAX_TURNS - 1:

        logger.info(
            "Approaching turn limit — forcing recommend"
        )

        return {
            **state,
            "intent": INTENT_RECOMMEND,
            "turn_count": turn_count,
        }

    latest_user = (
        _latest_user_message(messages)
        .lower()
    )

    # ======================================================
    # HARD OFF-TOPIC DETECTION
    # ======================================================

    if any(
        k in latest_user
        for k in SALARY_KEYWORDS
    ):

        return {
            **state,
            "intent": INTENT_OFF_TOPIC,
            "turn_count": turn_count,
        }

    # ======================================================
    # VERY VAGUE QUERY
    # ======================================================

    vague_inputs = [
        "i need an assessment",
        "need an assessment",
        "help me choose",
        "recommend an assessment",
    ]

    if (
        latest_user.strip()
        in vague_inputs
    ):

        return {
            **state,
            "intent": INTENT_CLARIFY,
            "turn_count": turn_count,
        }

    conversation = format_conversation(
        messages
    )

    prompt = intent_classification_prompt(
        conversation
    )

    raw = _llm_call(
        llm,
        prompt,
        system="",
    )

    intent = (
        raw
        .strip()
        .lower()
        .replace('"', "")
        .replace("'", "")
    )

    valid = {
        INTENT_CLARIFY,
        INTENT_RECOMMEND,
        INTENT_REFINE,
        INTENT_COMPARE,
        INTENT_OFF_TOPIC,
        INTENT_GREETING,
    }

    if intent not in valid:

        logger.warning(
            f"Unknown intent: {intent}"
        )

        intent = INTENT_CLARIFY

    return {
        **state,
        "intent": intent,
        "turn_count": turn_count,
    }


# ==========================================================
# EXTRACT FACTS
# ==========================================================

def extract_facts(
    state: AgentState,
    llm: ChatGroq,
) -> AgentState:

    conversation = format_conversation(
        state["messages"]
    )

    prompt = fact_extraction_prompt(
        conversation
    )

    raw = _llm_call(
        llm,
        prompt,
        system="",
    )

    parsed = _parse_json_response(raw)

    if not parsed:
        parsed = {}

    for field in [
        "what_to_measure",
        "specific_skills",
        "avoid_types",
        "assessment_preferences",
        "additional_constraints",
        "comparison_request",
    ]:

        if not isinstance(
            parsed.get(field),
            list,
        ):
            parsed[field] = []

    facts: ExtractedFacts = {
        "job_role": parsed.get(
            "job_role"
        ),

        "job_function": parsed.get(
            "job_function"
        ),

        "seniority": parsed.get(
            "seniority"
        ),

        "sector": parsed.get(
            "sector"
        ),

        "what_to_measure": parsed.get(
            "what_to_measure",
            [],
        ),

        "specific_skills": parsed.get(
            "specific_skills",
            [],
        ),

        "assessment_preferences": parsed.get(
            "assessment_preferences",
            [],
        ),

        "avoid_types": parsed.get(
            "avoid_types",
            [],
        ),

        "additional_constraints": parsed.get(
            "additional_constraints",
            [],
        ),

        "comparison_request": parsed.get(
            "comparison_request",
            [],
        ),

        "language_requirement": parsed.get(
            "language_requirement"
        ),

        "remote_required": parsed.get(
            "remote_required"
        ),

        "battery_stage": parsed.get(
            "battery_stage"
        ),
    }

    # ======================================================
    # HARD DOMAIN HEURISTICS
    # ======================================================

    latest_user = (
        _latest_user_message(
            state["messages"]
        ).lower()
    )

    # Leadership
    if any(
        t in latest_user
        for t in LEADERSHIP_TERMS
    ):

        if not facts.get("job_role"):
            facts["job_role"] = "Leadership"

        if "leadership" not in facts["what_to_measure"]:
            facts["what_to_measure"].append(
                "leadership"
            )

    # Safety / Industrial
    if any(
        t in latest_user
        for t in SAFETY_TERMS
    ):

        if not facts.get("sector"):
            facts["sector"] = "Manufacturing"

        for skill in [
            "safety",
            "dependability",
            "compliance",
            "reliability",
        ]:

            if skill not in facts["what_to_measure"]:
                facts["what_to_measure"].append(
                    skill
                )

        # Ensure job_role reflects safety context if not set
        if not facts.get("job_role"):
            facts["job_role"] = "Safety-Critical Role"

    # Contact Center
    if any(
        t in latest_user
        for t in CONTACT_CENTER_TERMS
    ):

        facts["job_role"] = (
            "Contact Center Agent"
        )

        for skill in [
            "customer service",
            "spoken english",
            "simulation",
        ]:

            if skill not in facts["what_to_measure"]:
                facts["what_to_measure"].append(
                    skill
                )

    # Healthcare
    if any(
        t in latest_user
        for t in HEALTHCARE_TERMS
    ):

        facts["sector"] = "Healthcare"

        for skill in [
            "hipaa",
            "medical terminology",
        ]:

            if skill not in facts["specific_skills"]:
                facts["specific_skills"].append(
                    skill
                )

    # Excel / Word / Admin
    if any(
        t in latest_user
        for t in OFFICE_TERMS
    ):

        for skill in [
            "excel",
            "word",
        ]:

            if skill not in facts["specific_skills"]:
                facts["specific_skills"].append(
                    skill
                )

    # ======================================================
    # PRESERVE PREVIOUS FACTS IN REFINEMENT
    # ======================================================

    previous_facts = state.get("facts", {})

    if state.get("intent") == INTENT_REFINE and previous_facts:

        # Merge: new facts override, but preserve previous lists
        for key in [
            "what_to_measure",
            "specific_skills",
            "assessment_preferences",
            "additional_constraints",
        ]:

            if key in previous_facts:
                prev_list = previous_facts.get(
                    key, []
                )

                curr_list = facts.get(
                    key, []
                )

                # Combine and deduplicate
                merged = list(
                    dict.fromkeys(
                        prev_list + curr_list
                    )
                )

                facts[key] = merged

        # Preserve role/sector if not changed
        if not facts.get("job_role"):
            facts["job_role"] = (
                previous_facts.get(
                    "job_role"
                )
            )

        if not facts.get("sector"):
            facts["sector"] = (
                previous_facts.get("sector")
            )

    return {
        **state,
        "facts": facts,
    }


# ==========================================================
# GENERATE CLARIFICATION
# ==========================================================

def generate_clarification(
    state: AgentState,
    llm: ChatGroq,
) -> AgentState:

    facts = state.get("facts", {})

    missing = []

    if (
        not facts.get("job_role")
        and not facts.get(
            "specific_skills"
        )
    ):
        missing.append("job_role")

    if (
        not facts.get(
            "what_to_measure"
        )
        and not facts.get(
            "specific_skills"
        )
    ):
        missing.append(
            "what_to_measure"
        )

    if not missing:

        logger.info(
            "Enough context exists — switching to recommend"
        )

        return {
            **state,
            "intent": INTENT_RECOMMEND,
        }

    conversation = format_conversation(
        state["messages"]
    )

    turns_remaining = (
        MAX_TURNS
        - len(state["messages"])
    )

    prompt = clarifying_question_prompt(
        conversation,
        missing,
        turns_remaining,
    )

    question = _llm_call(
        llm,
        prompt,
    )

    return {
        **state,
        "reply": question,
        "recommendations": [],
        "end_of_conversation": False,
    }


# ==========================================================
# RETRIEVE CANDIDATES
# ==========================================================

def retrieve_candidates(
    state: AgentState,
    vector_store,
) -> AgentState:

    facts = state.get("facts", {})

    query = _build_retrieval_query(
        facts
    )

    if not query:
        query = "assessment"

    logger.info(
        f"Retrieval query: {query}"
    )

    candidates = vector_store.retrieve(
        query=query,

        n_results=25,

        filter_remote=(
            True
            if facts.get(
                "remote_required"
            )
            else None
        ),
    )

    return {
        **state,
        "candidates": candidates,
    }


# ==========================================================
# RERANK + SELECT
# ==========================================================

def rerank_and_select(
    state: AgentState,
    llm: ChatGroq,
) -> AgentState:

    candidates = state.get(
        "candidates",
        [],
    )

    facts = state.get(
        "facts",
        {},
    )

    previous_shortlist = state.get(
        "shortlist",
        [],
    )

    latest_user = (
        _latest_user_message(
            state["messages"]
        ).lower()
    )

    avoid_types = {
        t.lower().strip()
        for t in facts.get(
            "avoid_types",
            []
        )
    }

    filtered = []

    for c in candidates:

        name = c.get(
            "name",
            "",
        ).lower()

        if (
            "opq32r" in name
            and (
                "personality"
                in avoid_types
                or "p"
                in avoid_types
            )
        ):
            continue

        filtered.append(c)

    if not filtered:
        filtered = candidates

    candidates_text = ""

    for i, c in enumerate(
        filtered[:20]
    ):

        candidates_text += (
            f"[{i}] "
            f"{c['name']} | "
            f"Type: {c.get('test_types','')} | "
            f"Duration: {c.get('duration','')} | "
            f"{(c.get('description','') or '')[:120]}\n"
        )

    prompt = rerank_prompt(
        facts,
        candidates_text,
        n=MAX_RECOMMENDATIONS,
    )

    raw = _llm_call(
        llm,
        prompt,
    )

    indices = None

    try:

        match = re.search(
            r"\[[\d,\s]+\]",
            raw,
            re.DOTALL,
        )

        if match:

            indices = json.loads(
                match.group(0)
            )

    except Exception:

        indices = None

    if not isinstance(indices, list):

        logger.warning(
            "Rerank parse failed — using fallback"
        )

        indices = []

        preferred = []

        for idx, c in enumerate(filtered):

            name = (
                c["name"]
                .lower()
            )

            test_type = c.get(
                "test_type",
                ""
            ).upper()

            # Leadership (P type)
            if (
                "leadership"
                in name
                or "opq"
                in name
                or test_type == "P"
            ):
                preferred.append(idx)

            # Safety (P type for DSI/dependability)
            elif (
                "safety"
                in name
                or "dependability"
                in name
                or "dsi"
                in name
                or (test_type == "P" and (
                    "manufacturing" in latest_user.lower()
                    or "plant" in latest_user.lower()
                ))
            ):
                preferred.append(idx)

            # Situational Judgement (B type)
            elif (
                "situational" in name
                or "scenario" in name
                or test_type == "B"
            ):
                preferred.append(idx)

            # Graduate/Cognitive
            elif (
                "graduate"
                in name
                or "verify"
                in name
                or test_type == "A"
            ):
                preferred.append(idx)

        if preferred:
            indices = preferred[:MAX_RECOMMENDATIONS]

        else:
            # Fallback: use first N by type diversity
            indices = list(
                range(
                    min(
                        MAX_RECOMMENDATIONS,
                        len(filtered),
                    )
                )
            )

    indices = [
        i
        for i in indices
        if (
            isinstance(i, int)
            and i < len(filtered)
        )
    ]

    shortlist = [
        filtered[i]
        for i in indices[
            :MAX_RECOMMENDATIONS
        ]
    ]

    # ======================================================
    # REFINEMENT CONTINUITY
    # ======================================================

    if (
        state.get("intent")
        == INTENT_REFINE
        and previous_shortlist
    ):

        # Preserve all previous items unless explicitly removed
        existing_names = {
            s["name"]
            for s in shortlist
        }

        for prev in previous_shortlist:

            if prev["name"] not in existing_names:
                shortlist.append(prev)

        # Ensure we have enough recommendations
        # Sort by position to maintain order
        shortlist = shortlist[:MAX_RECOMMENDATIONS]

    # ======================================================
    # DEDUPLICATION
    # ======================================================

    deduped = []

    seen = set()

    for s in shortlist:

        name = s["name"].strip()

        if name not in seen:
            deduped.append(s)
            seen.add(name)

    shortlist = deduped[
        :MAX_RECOMMENDATIONS
    ]

    logger.info(
        f"Final shortlist: "
        f"{[s['name'] for s in shortlist]}"
    )

    return {
        **state,
        "shortlist": shortlist,
    }


# ==========================================================
# GENERATE RECOMMENDATION REPLY
# ==========================================================

def generate_recommendation_reply(
    state: AgentState,
    llm: ChatGroq,
) -> AgentState:

    facts = state.get("facts", {})

    shortlist = state.get(
        "shortlist",
        [],
    )

    is_refinement = (
        state.get("intent")
        == INTENT_REFINE
    )

    prompt = recommendation_reply_prompt(
        facts,
        shortlist,
        is_refinement,
    )

    reply = _llm_call(
        llm,
        prompt,
    )

    return {
        **state,
        "reply": reply,
        "end_of_conversation":
            _user_has_confirmed(
                state["messages"]
            ),
    }


# ==========================================================
# COMPARE
# ==========================================================

def compare_assessments(
    state: AgentState,
    llm: ChatGroq,
    vector_store,
) -> AgentState:

    facts = state.get("facts", {})

    requested = facts.get(
        "comparison_request",
        [],
    )

    matched = []

    if requested:

        matched = (
            vector_store.get_by_names(
                requested
            )
        )

    if not matched:

        last_user = ""

        for m in reversed(
            state["messages"]
        ):

            if m["role"] == "user":
                last_user = m[
                    "content"
                ]
                break

        matched = vector_store.retrieve(
            last_user,
            n_results=3,
        )

    conversation = format_conversation(
        state["messages"]
    )

    prompt = comparison_reply_prompt(
        facts,
        matched,
        conversation,
    )

    reply = _llm_call(
        llm,
        prompt,
    )

    return {
        **state,
        "reply": reply,
        "shortlist": [],
        "end_of_conversation": False,
    }


# ==========================================================
# REFUSE
# ==========================================================

def refuse(
    state: AgentState,
    llm: ChatGroq,
) -> AgentState:

    last_user = ""

    for m in reversed(
        state["messages"]
    ):

        if m["role"] == "user":
            last_user = m[
                "content"
            ]
            break

    prompt = off_topic_reply_prompt(
        last_user
    )

    reply = _llm_call(
        llm,
        prompt,
    )

    return {
        **state,
        "reply": reply,
        "shortlist": [],
        "end_of_conversation": False,
    }


# ==========================================================
# GREETING
# ==========================================================

def greet(
    state: AgentState,
    llm: ChatGroq,
) -> AgentState:

    reply = _llm_call(
        llm,
        greeting_reply_prompt(),
    )

    return {
        **state,
        "reply": reply,
        "shortlist": [],
        "end_of_conversation": False,
    }