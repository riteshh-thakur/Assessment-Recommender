"""
SHL Evaluation Harness
----------------------

Evaluator aligned with SHL trace behavior.

Measures:
- Recall@K
- Hallucination rate
- Recommendation count compliance
- end_of_conversation correctness
- Off-topic refusal behavior
- Comparison behavior
- Average recommendation count
- Turn efficiency
- Schema compliance
- Vague query clarification
- Empty recommendation behavior
- Refinement continuity
- Turn limit compliance

Usage:
    python -m scripts.evaluate --traces data/traces.json
"""

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

BASE_URL = os.getenv(
    "EVAL_BASE_URL",
    "http://localhost:8000",
)

MAX_ALLOWED_RECOMMENDATIONS = 10
MAX_TURNS = 8


# ==========================================================
# RECALL
# ==========================================================

def recall_at_k(
    recommended_names: list[str],
    relevant_names: list[str],
    k: int = 10,
) -> float:

    if not relevant_names:
        return 1.0

    recommended = {
        r.lower().strip()
        for r in recommended_names[:k]
    }

    relevant = {
        r.lower().strip()
        for r in relevant_names
    }

    matched = 0

    for rel in relevant:

        for rec in recommended:

            if (
                rel in rec
                or rec in rel
            ):
                matched += 1
                break

    return matched / len(relevant)


# ==========================================================
# HALLUCINATION CHECK
# ==========================================================

def hallucination_rate(
    recommended_names: list[str],
    catalog_names: set[str],
) -> float:

    if not recommended_names:
        return 0.0

    hallucinated = 0

    for name in recommended_names:

        lower = name.lower().strip()

        matched = any(
            lower == c.lower().strip()
            for c in catalog_names
        )

        if not matched:
            hallucinated += 1

    return hallucinated / len(
        recommended_names
    )


# ==========================================================
# LOAD CATALOG NAMES
# ==========================================================

def load_catalog_names(
    catalog_path: str = "data/catalog.json",
) -> set[str]:

    with open(
        catalog_path,
        encoding="utf-8",
    ) as f:

        catalog = json.load(f)

    return {
        a["name"]
        for a in catalog
        if a.get("name")
    }


# ==========================================================
# RUN TRACE
# ==========================================================

def run_conversation(
    trace: dict,
) -> dict:

    conversation = trace["conversation"]

    try:

        response = httpx.post(
            f"{BASE_URL}/chat",
            json={
                "messages": conversation
            },
            timeout=60.0,
        )

        response.raise_for_status()

        data = response.json()

        recommendations = data.get(
            "recommendations",
            [],
        )

        return {
            "reply": data.get(
                "reply",
                "",
            ),

            "recommendations":
                recommendations,

            "recommended_names": [
                r.get("name", "")
                for r in recommendations
            ],

            "end_of_conversation":
                data.get(
                    "end_of_conversation",
                    False,
                ),
        }

    except Exception as e:

        logger.error(
            f"Trace failed: "
            f"{trace.get('trace_id')} "
            f"{e}"
        )

        return {
            "reply": "",
            "recommendations": [],
            "recommended_names": [],
            "end_of_conversation": False,
            "error": str(e),
        }


# ==========================================================
# SCHEMA COMPLIANCE
# ==========================================================

def check_schema_compliance(
    result: dict,
) -> tuple[bool, list[str]]:

    errors = []

    if "reply" not in result:
        errors.append("Missing reply")

    if "recommendations" not in result:
        errors.append("Missing recommendations")

    if "end_of_conversation" not in result:
        errors.append(
            "Missing end_of_conversation"
        )

    if not isinstance(
        result.get("reply"),
        str,
    ):
        errors.append(
            "reply must be string"
        )

    if not isinstance(
        result.get("recommendations"),
        list,
    ):
        errors.append(
            "recommendations must be list"
        )

    if not isinstance(
        result.get(
            "end_of_conversation"
        ),
        bool,
    ):
        errors.append(
            "end_of_conversation must be bool"
        )

    for rec in result.get(
        "recommendations",
        [],
    ):

        if "name" not in rec:
            errors.append(
                "Recommendation missing name"
            )

        if "url" not in rec:
            errors.append(
                "Recommendation missing url"
            )

        if (
            "test_type" not in rec
            and "test_types" not in rec
        ):
            errors.append(
                "Recommendation missing type"
            )

        url = rec.get("url", "")

        if (
            "shl.com"
            not in url.lower()
        ):
            errors.append(
                f"Invalid SHL URL: {url}"
            )

    return (
        len(errors) == 0,
        errors,
    )


# ==========================================================
# HEURISTIC CHECKS
# ==========================================================

def check_end_behavior(
    trace: dict,
    result: dict,
) -> bool:

    last_user = ""

    for m in reversed(
        trace["conversation"]
    ):

        if m["role"] == "user":
            last_user = (
                m["content"]
                .lower()
            )
            break

    confirmation_phrases = [
        "confirmed",
        "that works",
        "perfect",
        "that covers it",
        "final list",
        "keep the shortlist",
        "that's good",
    ]

    expected = any(
        p in last_user
        for p in confirmation_phrases
    )

    return (
        expected
        == result[
            "end_of_conversation"
        ]
    )


def check_recommendation_limit(
    result: dict,
) -> bool:

    return (
        len(
            result[
                "recommendations"
            ]
        )
        <= MAX_ALLOWED_RECOMMENDATIONS
    )


def check_comparison_behavior(
    trace: dict,
    result: dict,
) -> bool:

    comparison_keywords = [
        "difference between",
        "compare",
        "different from",
    ]

    conversation_text = " ".join([
        m["content"].lower()
        for m in trace[
            "conversation"
        ]
    ])

    is_comparison = any(
        k in conversation_text
        for k in comparison_keywords
    )

    if not is_comparison:
        return True

    return (
        len(
            result[
                "recommendations"
            ]
        )
        == 0
    )


def check_off_topic_behavior(
    trace: dict,
    result: dict,
) -> bool:

    off_topic_keywords = [
        "legally required",
        "legal requirement",
        "compliance obligation",
    ]

    text = " ".join([
        m["content"].lower()
        for m in trace[
            "conversation"
        ]
    ])

    off_topic = any(
        k in text
        for k in off_topic_keywords
    )

    if not off_topic:
        return True

    return (
        result[
            "end_of_conversation"
        ]
        is False
    )


# ==========================================================
# VAGUE QUERY CHECK
# ==========================================================

def check_vague_query_behavior(
    trace: dict,
    result: dict,
) -> bool:

    vague_inputs = [
        "i need an assessment",
        "help me choose",
        "need a test",
    ]

    first_user = ""

    for m in trace["conversation"]:

        if m["role"] == "user":
            first_user = (
                m["content"]
                .lower()
                .strip()
            )
            break

    is_vague = any(
        v == first_user
        for v in vague_inputs
    )

    if not is_vague:
        return True

    return (
        len(
            result[
                "recommendations"
            ]
        )
        == 0
    )


# ==========================================================
# EMPTY RECOMMENDATION CHECK
# ==========================================================

def check_empty_recommendation_behavior(
    trace: dict,
    result: dict,
) -> bool:

    first_user = ""

    for m in trace["conversation"]:

        if m["role"] == "user":
            first_user = (
                m["content"]
                .lower()
            )
            break

    vague = (
        "assessment" in first_user
        and len(first_user.split()) <= 5
    )

    if vague:

        return (
            len(
                result[
                    "recommendations"
                ]
            )
            == 0
        )

    return True


# ==========================================================
# TURN LIMIT CHECK
# ==========================================================

def check_turn_limit(
    trace: dict,
) -> bool:

    return (
        len(trace["conversation"])
        <= MAX_TURNS
    )


# ==========================================================
# REFINEMENT CONTINUITY CHECK
# ==========================================================

def check_refinement_behavior(
    trace: dict,
    result: dict,
) -> bool:

    text = " ".join([
        m["content"].lower()
        for m in trace[
            "conversation"
        ]
    ])

    refinement_keywords = [
        "add",
        "remove",
        "replace",
        "drop",
    ]

    is_refinement = any(
        k in text
        for k in refinement_keywords
    )

    if not is_refinement:
        return True

    return (
        len(
            result[
                "recommendations"
            ]
        )
        > 0
    )


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate(
    traces_path: str,
    k: int = 10,
):

    with open(
        traces_path,
        encoding="utf-8",
    ) as f:

        traces = json.load(f)

    catalog_names = (
        load_catalog_names()
    )

    logger.info(
        f"Evaluating "
        f"{len(traces)} traces"
    )

    recall_scores = []
    hallucination_scores = []
    recommendation_counts = []

    end_behavior_checks = []
    comparison_checks = []
    off_topic_checks = []
    limit_checks = []

    schema_checks = []
    vague_checks = []
    empty_rec_checks = []
    turn_checks = []
    refinement_checks = []

    results = []

    for trace in traces:

        trace_id = trace.get(
            "trace_id",
            "unknown",
        )

        logger.info(
            f"Running trace: {trace_id}"
        )

        result = run_conversation(
            trace
        )

        recommended = result[
            "recommended_names"
        ]

        relevant = trace.get(
            "relevant_assessments",
            [],
        )

        recall = recall_at_k(
            recommended,
            relevant,
            k=k,
        )

        hallucination = (
            hallucination_rate(
                recommended,
                catalog_names,
            )
        )

        recall_scores.append(
            recall
        )

        hallucination_scores.append(
            hallucination
        )

        recommendation_counts.append(
            len(recommended)
        )

        end_ok = check_end_behavior(
            trace,
            result,
        )

        comparison_ok = (
            check_comparison_behavior(
                trace,
                result,
            )
        )

        off_topic_ok = (
            check_off_topic_behavior(
                trace,
                result,
            )
        )

        limit_ok = (
            check_recommendation_limit(
                result
            )
        )

        schema_ok, schema_errors = (
            check_schema_compliance(
                result
            )
        )

        vague_ok = (
            check_vague_query_behavior(
                trace,
                result,
            )
        )

        empty_ok = (
            check_empty_recommendation_behavior(
                trace,
                result,
            )
        )

        turn_ok = (
            check_turn_limit(
                trace
            )
        )

        refinement_ok = (
            check_refinement_behavior(
                trace,
                result,
            )
        )

        end_behavior_checks.append(
            end_ok
        )

        comparison_checks.append(
            comparison_ok
        )

        off_topic_checks.append(
            off_topic_ok
        )

        limit_checks.append(
            limit_ok
        )

        schema_checks.append(
            schema_ok
        )

        vague_checks.append(
            vague_ok
        )

        empty_rec_checks.append(
            empty_ok
        )

        turn_checks.append(
            turn_ok
        )

        refinement_checks.append(
            refinement_ok
        )

        results.append({
            "trace_id": trace_id,

            "recommended":
                recommended,

            "relevant":
                relevant,

            f"recall@{k}":
                round(recall, 4),

            "hallucination_rate":
                round(
                    hallucination,
                    4,
                ),

            "recommendation_count":
                len(recommended),

            "schema_compliant":
                schema_ok,

            "schema_errors":
                schema_errors,

            "end_behavior_correct":
                end_ok,

            "comparison_behavior_correct":
                comparison_ok,

            "off_topic_behavior_correct":
                off_topic_ok,

            "limit_respected":
                limit_ok,

            "vague_query_behavior_correct":
                vague_ok,

            "empty_recommendation_behavior_correct":
                empty_ok,

            "turn_limit_respected":
                turn_ok,

            "refinement_behavior_correct":
                refinement_ok,
        })

        logger.info(
            f"Recall@{k}: {recall:.3f} | "
            f"Hallucination: {hallucination:.3f}"
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    mean_recall = (
        statistics.mean(
            recall_scores
        )
        if recall_scores
        else 0.0
    )

    mean_hallucination = (
        statistics.mean(
            hallucination_scores
        )
        if hallucination_scores
        else 0.0
    )

    avg_recommendations = (
        statistics.mean(
            recommendation_counts
        )
        if recommendation_counts
        else 0.0
    )

    print("\n" + "=" * 60)

    print("SHL EVALUATION RESULTS")

    print("=" * 60)

    print(
        f"Traces evaluated: "
        f"{len(traces)}"
    )

    print(
        f"Mean Recall@{k}: "
        f"{mean_recall:.4f}"
    )

    print(
        f"Mean Hallucination Rate: "
        f"{mean_hallucination:.4f}"
    )

    print(
        f"Average Recommendation Count: "
        f"{avg_recommendations:.2f}"
    )

    print(
        f"Schema Compliance: "
        f"{sum(schema_checks)}/{len(schema_checks)}"
    )

    print(
        f"End Behavior Accuracy: "
        f"{sum(end_behavior_checks)}/{len(end_behavior_checks)}"
    )

    print(
        f"Comparison Behavior Accuracy: "
        f"{sum(comparison_checks)}/{len(comparison_checks)}"
    )

    print(
        f"Off-topic Behavior Accuracy: "
        f"{sum(off_topic_checks)}/{len(off_topic_checks)}"
    )

    print(
        f"Recommendation Limit Compliance: "
        f"{sum(limit_checks)}/{len(limit_checks)}"
    )

    print(
        f"Vague Query Handling Accuracy: "
        f"{sum(vague_checks)}/{len(vague_checks)}"
    )

    print(
        f"Empty Recommendation Accuracy: "
        f"{sum(empty_rec_checks)}/{len(empty_rec_checks)}"
    )

    print(
        f"Turn Limit Compliance: "
        f"{sum(turn_checks)}/{len(turn_checks)}"
    )

    print(
        f"Refinement Continuity Accuracy: "
        f"{sum(refinement_checks)}/{len(refinement_checks)}"
    )

    print("=" * 60)

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    output = {
        "mean_recall": mean_recall,

        "mean_hallucination":
            mean_hallucination,

        "average_recommendations":
            avg_recommendations,

        "schema_compliance":
            sum(schema_checks),

        "traces": results,
    }

    output_path = Path(
        "data/eval_results.json"
    )

    output_path.parent.mkdir(
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    logger.info(
        f"Saved results to "
        f"{output_path}"
    )

    return output


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--traces",
        default="data/traces.json",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--url",
        default=None,
    )

    args = parser.parse_args()

    global BASE_URL

    if args.url:
        BASE_URL = args.url

    evaluate(
        traces_path=args.traces,
        k=args.k,
    )


if __name__ == "__main__":
    main()