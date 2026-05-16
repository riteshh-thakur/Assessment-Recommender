"""
Agent State
-----------
TypedDicts for LangGraph state management.

Updated using:
- SHL evaluator traces
- refinement behavior
- shortlist continuity
- layered battery recommendations
- comparison handling
- evaluator-aligned metadata fields
"""

from typing import Optional, TypedDict
from pydantic import BaseModel


# ==========================================================
# INTENT CONSTANTS
# ==========================================================

INTENT_CLARIFY = "clarify"
INTENT_RECOMMEND = "recommend"
INTENT_REFINE = "refine"
INTENT_COMPARE = "compare"
INTENT_OFF_TOPIC = "off_topic"
INTENT_GREETING = "greeting"


# ==========================================================
# LIMITS
# ==========================================================

MAX_TURNS = 12
MAX_RECOMMENDATIONS = 10


# ==========================================================
# CLARIFICATION PRIORITY
# ==========================================================

CLARIFICATION_PRIORITY = [

    (
        "job_role",
        (
            "What specific role or job title "
            "are you hiring for?"
        )
    ),

    (
        "specific_skills",
        (
            "Are there specific skills, "
            "technologies, or tools "
            "you want assessed?"
        )
    ),

    (
        "what_to_measure",
        (
            "What would you mainly like "
            "to assess — technical skills, "
            "cognitive ability, personality, "
            "situational judgement, or a mix?"
        )
    ),

    (
        "seniority",
        (
            "What seniority level is this role "
            "— graduate, entry-level, "
            "mid-level, senior, or executive?"
        )
    ),

    (
        "language_requirement",
        (
            "Do the assessments need to "
            "support a specific language?"
        )
    ),

    (
        "remote_required",
        (
            "Do you need assessments that "
            "can be administered remotely?"
        )
    ),
]


# ==========================================================
# EXTRACTED FACTS
# ==========================================================

class ExtractedFacts(
    TypedDict,
    total=False,
):

    # ------------------------------------------------------
    # ROLE / DOMAIN
    # ------------------------------------------------------

    job_role: Optional[str]

    job_function: Optional[str]

    seniority: Optional[str]

    sector: Optional[str]

    # ------------------------------------------------------
    # ASSESSMENT TARGETS
    # ------------------------------------------------------

    what_to_measure: list[str]

    specific_skills: list[str]

    # ------------------------------------------------------
    # USER PREFERENCES
    # ------------------------------------------------------

    assessment_preferences: list[str]

    avoid_types: list[str]

    additional_constraints: list[str]

    # ------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------

    comparison_request: list[str]

    # ------------------------------------------------------
    # DELIVERY / LANGUAGE
    # ------------------------------------------------------

    language_requirement: Optional[str]

    remote_required: Optional[bool]

    # ------------------------------------------------------
    # BATTERY CONTEXT
    # ------------------------------------------------------

    battery_stage: Optional[str]


# ==========================================================
# AGENT STATE
# ==========================================================

class AgentState(
    TypedDict,
    total=False,
):

    # ------------------------------------------------------
    # CONVERSATION
    # ------------------------------------------------------

    messages: list[dict]

    turn_count: int

    agent_turn_count: int

    # ------------------------------------------------------
    # INTENT
    # ------------------------------------------------------

    intent: Optional[str]

    # ------------------------------------------------------
    # FACTS
    # ------------------------------------------------------

    facts: ExtractedFacts

    # ------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------

    candidates: list[dict]

    shortlist: list[dict]

    # ------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------

    clarifying_question: Optional[str]

    reply: str

    recommendations: list[dict]

    end_of_conversation: bool


# ==========================================================
# API MODELS
# ==========================================================

class Message(BaseModel):

    role: str

    content: str


class ChatRequest(BaseModel):

    messages: list[Message]


class Recommendation(BaseModel):

    name: str

    url: str

    # backward compatible
    test_type: Optional[str] = None

    # preferred field
    test_types: Optional[str] = None

    duration: Optional[str] = None

    description: Optional[str] = None


class ChatResponse(BaseModel):

    reply: str

    recommendations: list[Recommendation]

    end_of_conversation: bool