"""
Prompt Templates
----------------
Centralized prompts tuned against:
- SHL evaluator traces
- Recall@10 optimization
- Conversational refinement behavior
- Grounded comparison behavior
- Catalog-safe recommendation generation
"""

from typing import Optional


# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are an SHL Assessment Advisor.

You help recruiters and hiring managers choose SHL assessments from the official SHL product catalog.

You ONLY discuss:
- SHL assessments
- SHL reports
- SHL assessment batteries
- SHL assessment comparisons

You DO NOT:
- give legal advice
- interpret regulations
- discuss compensation
- recommend competitor products
- invent assessments
- hallucinate URLs
- answer unrelated questions

==========================================================
IMPORTANT BEHAVIOR RULES
==========================================================

1. RECOMMEND EARLY
If the user provides:
- a job role,
- a JD,
- a hiring use case,
- or a domain,

you should usually recommend immediately.

Do NOT over-clarify.

==========================================================

2. ASK ONLY ONE QUESTION
If clarification is necessary:
- ask ONE high-value question only
- prioritize questions that significantly change recommendations

==========================================================

3. OPQ32r DEFAULT RULE
For:
- leadership,
- management,
- senior IC,
- executive,
- customer-facing,
- graduate programs,
- sales,
- professional roles,

OPQ32r is usually included as the default personality layer unless the user explicitly removes it.

==========================================================

4. REFINEMENT BEHAVIOR
Users may:
- add tests
- remove tests
- replace tests
- shorten the battery
- request simulations
- request cognitive tests
- request language changes

You MUST:
- preserve previous valid recommendations
- modify incrementally
- avoid rebuilding from scratch

==========================================================

5. COMPARISON BEHAVIOR
Comparison replies:
- must stay grounded in catalog data
- should explain differences clearly
- should explain relationships between instruments and reports
- should NOT generate a new shortlist unless user asks

==========================================================

6. CATALOG LIMITATIONS
If SHL does not have:
- a direct technology test,
- a language variant,
- or an exact match,

say so honestly and suggest nearest alternatives.

==========================================================

7. LEGAL / COMPLIANCE QUESTIONS
If the user asks:
- legal,
- regulatory,
- compliance interpretation,

politely refuse that part only.

Continue helping with assessment selection.

==========================================================

8. END OF CONVERSATION
Set end_of_conversation=true ONLY when the user:
- confirms,
- locks in,
- finalizes,
- says "that works",
- says "confirmed",
- says "perfect",
- says "that covers it".

==========================================================

TONE
==========================================================

- concise
- professional
- conversational
- grounded
- no marketing fluff
- no generic AI phrasing
"""


# ==========================================================
# INTENT CLASSIFICATION
# ==========================================================

def intent_classification_prompt(
    conversation: str,
) -> str:

    return f"""
Analyze the user's CURRENT intent.

CONVERSATION:
{conversation}

Classify into EXACTLY ONE:

- recommend
- refine
- compare
- clarify
- off_topic
- greeting

==========================================================
INTENT DEFINITIONS
==========================================================

recommend:
- user provided role/domain/JD/use case
- enough context exists to recommend
- LEAN STRONGLY toward recommend

Examples:
- "Hiring Java developers"
- "Need assessments for sales managers"
- pasted JD
- "Need graduate screening"

----------------------------------------------------------

refine:
User modifies existing shortlist.

Examples:
- "Add AWS"
- "Remove OPQ"
- "Add simulations"
- "Drop cognitive tests"
- "Replace with shorter option"

----------------------------------------------------------

compare:
User compares assessments.

Examples:
- "Difference between OPQ and MQ?"
- "How is DSI different?"

----------------------------------------------------------

clarify:
ONLY if request is extremely vague.

Examples:
- "Need an assessment"
- "Help me choose"

WITHOUT role/domain/context.

----------------------------------------------------------

off_topic:
Legal/regulatory/general HR advice.

Examples:
- HIPAA legal interpretation
- compensation
- hiring law

----------------------------------------------------------

greeting:
Hello/small talk only.

==========================================================
IMPORTANT RULES
==========================================================

- If ANY role/domain/use-case exists → recommend
- Leadership/executive requests count as role context
- Safety/manufacturing requests count as role context
- Contact-center/customer-service requests count as role context
- Healthcare/HIPAA requests count as role context
- Excel/Word/admin requests count as role context
- JD → recommend immediately
- Add/remove/change → refine
- Comparison request → compare
- Legal interpretation → off_topic

Return ONLY the intent word.
"""


# ==========================================================
# FACT EXTRACTION
# ==========================================================

def fact_extraction_prompt(
    conversation: str,
) -> str:

    return f"""
Extract structured hiring facts from this conversation.

Return ONLY valid JSON.

CONVERSATION:
{conversation}

JSON FORMAT:
{{
  "job_role": null,
  "job_function": null,
  "seniority": null,
  "what_to_measure": [],
  "specific_skills": [],
  "assessment_preferences": [],
  "avoid_types": [],
  "additional_constraints": [],
  "language_requirement": null,
  "sector": null,
  "comparison_request": [],
  "remote_required": null,
  "battery_stage": null
}}

==========================================================
EXTRACTION RULES
==========================================================

job_role:
Specific role title.

Examples:
- Java Developer
- Sales Manager
- Graduate Trainee
- Plant Operator

----------------------------------------------------------

what_to_measure:
Examples:
- cognitive ability
- personality
- technical skills
- situational judgement
- safety behaviour
- spoken language

----------------------------------------------------------

specific_skills:
Examples:
- Java
- Spring
- SQL
- AWS
- Docker
- Excel
- HIPAA
- Rust

----------------------------------------------------------

assessment_preferences:
Explicit preferences.

Examples:
- simulations
- short assessments
- high-volume screening
- leadership benchmark

----------------------------------------------------------

avoid_types:
Examples:
- personality tests
- simulations
- long tests

----------------------------------------------------------

battery_stage:
Infer if user is describing:
- first-stage screening
- finalist-stage depth
- development
- leadership benchmarking

----------------------------------------------------------

DOMAIN INFERENCE RULES

Leadership/executive/CXO/director:
- infer leadership assessment needs
- often implies OPQ32r

Safety/industrial/chemical/manufacturing:
- infer safety behaviour and dependability

Contact center/customer service/inbound calls:
- infer spoken language and simulation needs

Healthcare/HIPAA/patient records:
- infer healthcare administration context

Excel/Word/admin assistant:
- infer Microsoft Office capability testing

Graduate hiring:
- often implies cognitive ability and situational judgement

----------------------------------------------------------

IMPORTANT
==========================================================

- Use ENTIRE conversation history
- Latest refinement overrides earlier requests
- Infer obvious needs from context
- Graduate hiring usually implies cognitive testing
- Leadership usually implies OPQ32r
- Safety-critical roles imply DSI-style measures

Return ONLY valid JSON.
"""


# ==========================================================
# CLARIFICATION PROMPT
# ==========================================================

CLARIFICATION_PRIORITY = [
    (
        "job_role",
        "What specific role are you hiring for?"
    ),

    (
        "specific_skills",
        "Are there specific skills or technologies you want assessed?"
    ),

    (
        "seniority",
        "What seniority level is the role?"
    ),

    (
        "language_requirement",
        "Do the assessments need to support a specific language?"
    ),

    (
        "what_to_measure",
        "Are you mainly looking to assess technical skills, cognitive ability, personality, or a mix?"
    ),
]


def clarifying_question_prompt(
    conversation: str,
    missing_fields: list[str],
    turn_budget: int,
) -> str:

    return f"""
Generate ONE clarifying question.

CONVERSATION:
{conversation}

MISSING:
{missing_fields}

TURNS REMAINING:
{turn_budget}

RULES:
- Ask ONE question only
- Ask highest-impact question
- Avoid repeating prior questions
- Prefer recommending over over-clarifying
- Keep conversational tone
- Short and natural

Return ONLY the question.
"""


# ==========================================================
# RERANK PROMPT
# ==========================================================

def rerank_prompt(
    facts: dict,
    candidates_text: str,
    n: int = 10,
) -> str:

    return f"""
Rank the best SHL assessments.

HIRING NEED:
{_format_facts(facts)}

CANDIDATES:
{candidates_text}

==========================================================
RANKING RULES
==========================================================

1. PRIORITIZE DIRECT SKILL MATCH
If skills are named:
- Java
- SQL
- AWS
- Docker
- Excel
- HIPAA

then direct skill tests rank highest.

----------------------------------------------------------

2. PRESERVE REFINEMENTS
If user:
- added tests,
- removed tests,
- requested simulations,
- requested shorter batteries,

honor those changes STRICTLY.

----------------------------------------------------------

3. OPQ32r DEFAULT
Include OPQ32r for:
- leadership
- sales
- graduate programs
- senior IC
- executive
- customer-facing

UNLESS explicitly removed.

----------------------------------------------------------

4. VERIFY G+ RULE
Strongly consider Verify G+ for:
- senior technical roles
- graduate programs
- analytical/problem-solving roles

----------------------------------------------------------

5. SAFETY RULE
For:
- industrial
- manufacturing
- safety-critical
- plant operators
- chemical facility

MUST prioritize:
- DSI (Dependability & Safety Index)
- Safety & Dependability assessment

This is non-negotiable for safety-critical roles.

----------------------------------------------------------

6. SITUATIONAL JUDGEMENT
If user requests:
- "situational judgement"
- "scenario"
- "work context"
- "decision making"

Include Situational Judgement Test (type B).

----------------------------------------------------------

7. SIMULATION RULE
If user wants:
- practical capability
- real-world interaction
- volume screening
- live coding

favor simulations.

----------------------------------------------------------

8. LANGUAGE RULE
Respect language constraints strictly.

----------------------------------------------------------

9. TYPE DIVERSITY
Include variety:
- At least one cognitive (A/K)
- At least one personality (P) if leadership/safety
- At least one situational (B) if requested
- Avoid redundancy

----------------------------------------------------------

10. CATALOG LIMITATIONS
If no exact match exists:
- use nearest proxy
- prefer domain-adjacent assessments

==========================================================
OUTPUT FORMAT
==========================================================

Return ONLY valid JSON.

DO NOT:
- add markdown
- add explanations
- add comments
- add prose
- add code fences

VALID EXAMPLE:
[3,1,7,2]

INVALID:
```json
[3,1,7]
```

Return ONLY the JSON array.
"""


# ==========================================================
# RECOMMENDATION REPLY
# ==========================================================

def recommendation_reply_prompt(
    facts: dict,
    shortlist: list[dict],
    is_refinement: bool = False,
) -> str:

    recommendations = "\n".join([
        f"- {a['name']}"
        for a in shortlist
    ])

    return f"""
Generate a concise conversational recommendation reply.

HIRING NEED:
{_format_facts(facts)}

SHORTLIST:
{recommendations}

REFINEMENT:
{is_refinement}

RULES:
- 2-4 sentences only
- concise
- explain WHY top recommendations fit
- mention changes if refinement occurred
- do NOT repeat full table contents
- grounded and practical tone
- no fluff
- no fake enthusiasm

Return ONLY the reply.
"""


# ==========================================================
# COMPARISON PROMPT
# ==========================================================

def comparison_reply_prompt(
    facts: dict,
    assessments: list[dict],
    conversation: str,
) -> str:

    assessment_details = "\n\n".join([
        (
            f"NAME: {a['name']}\n"
            f"TYPE: {a.get('test_types', '')}\n"
            f"DURATION: {a.get('duration', '')}\n"
            f"DESCRIPTION: {a.get('description', '')}"
        )
        for a in assessments
    ])

    return f"""
Answer the comparison question using ONLY catalog data.

CONVERSATION:
{conversation}

ASSESSMENTS:
{assessment_details}

RULES:
- grounded only
- explain differences clearly
- explain relationships between reports/instruments
- concise
- no hallucinations
- no invented features

Return ONLY the comparison reply.
"""


# ==========================================================
# OFF TOPIC
# ==========================================================

def off_topic_reply_prompt(
    last_message: str,
) -> str:

    return f"""
User message:
{last_message}

Politely refuse legal/compliance/off-topic advice.

RULES:
- concise
- polite
- redirect toward SHL assessment selection
- do NOT end conversation

Return ONLY the reply.
"""


# ==========================================================
# GREETING
# ==========================================================

def greeting_reply_prompt() -> str:

    return """
Write a short greeting as an SHL Assessment Advisor.

Ask:
- what role they are hiring for
OR
- what they want assessed.

Maximum 2 sentences.

Return ONLY the reply.
"""


# ==========================================================
# CATALOG GAP
# ==========================================================

def catalog_gap_reply_prompt(
    missing_skill: str,
    conversation: str,
) -> str:

    return f"""
The catalog does not contain a direct test for:
{missing_skill}

CONVERSATION:
{conversation}

Write a concise response that:
- acknowledges the gap honestly
- suggests closest alternatives
- offers to build a shortlist

Return ONLY the reply.
"""


# ==========================================================
# HELPERS
# ==========================================================

def _format_facts(
    facts: dict,
) -> str:

    lines = []

    if facts.get("job_role"):
        lines.append(
            f"Role: {facts['job_role']}"
        )

    if facts.get("job_function"):
        lines.append(
            f"Function: {facts['job_function']}"
        )

    if facts.get("seniority"):
        lines.append(
            f"Seniority: {facts['seniority']}"
        )

    if facts.get("sector"):
        lines.append(
            f"Sector: {facts['sector']}"
        )

    if facts.get("what_to_measure"):
        lines.append(
            "Assess: "
            + ", ".join(
                facts["what_to_measure"]
            )
        )

    if facts.get("specific_skills"):
        lines.append(
            "Skills: "
            + ", ".join(
                facts["specific_skills"]
            )
        )

    if facts.get("language_requirement"):
        lines.append(
            f"Language: "
            f"{facts['language_requirement']}"
        )

    if (
        facts.get("remote_required")
        is not None
    ):
        lines.append(
            f"Remote: "
            f"{facts['remote_required']}"
        )

    if facts.get("avoid_types"):
        lines.append(
            "Avoid: "
            + ", ".join(
                facts["avoid_types"]
            )
        )

    if facts.get("additional_constraints"):
        lines.append(
            "Constraints: "
            + ", ".join(
                facts[
                    "additional_constraints"
                ]
            )
        )

    return (
        "\n".join(lines)
        if lines
        else "No structured facts extracted."
    )


def format_conversation(
    messages: list[dict],
) -> str:

    lines = []

    for m in messages:

        role = (
            "User"
            if m["role"] == "user"
            else "Assistant"
        )

        lines.append(
            f"{role}: {m['content']}"
        )

    return "\n".join(lines)