"""
Enhanced SHL Vector Store
-------------------------
Optimized for:
- SHL conversational evaluator traces
- Recall@10
- Hybrid retrieval
- Role-aware ranking
- Comparison grounding
- Refinement-friendly retrieval

Key improvements:
- Semantic + keyword hybrid scoring
- Rich search text construction
- Better metadata handling
- Domain-aware retrieval
- Deterministic ranking before LLM rerank
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "shl_assessments"


class SHLVectorStore:
    def __init__(
        self,
        catalog_path: str,
        chroma_persist_dir: str,
    ):
        self.catalog_path = Path(catalog_path)

        self.persist_dir = Path(chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")

        self.embedder = SentenceTransformer(
            EMBED_MODEL_NAME
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False
            ),
        )

        self.collection = None

    # ==========================================================
    # BUILD INDEX
    # ==========================================================

    def build_index(
        self,
        force_rebuild: bool = False,
    ) -> None:

        existing = [
            c.name
            for c in self.client.list_collections()
        ]

        if (
            COLLECTION_NAME in existing
            and not force_rebuild
        ):
            logger.info(
                "Loading existing collection..."
            )

            self.collection = self.client.get_collection(
                COLLECTION_NAME
            )

            logger.info(
                f"Loaded {self.collection.count()} assessments"
            )

            return

        if COLLECTION_NAME in existing:
            self.client.delete_collection(
                COLLECTION_NAME
            )

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {self.catalog_path}"
            )

        with open(
            self.catalog_path,
            "r",
            encoding="utf-8",
        ) as f:
            assessments = json.load(f)

        logger.info(
            f"Building embeddings for "
            f"{len(assessments)} assessments"
        )

        ids = []
        docs = []
        embeddings = []
        metadatas = []

        for idx, assessment in enumerate(
            assessments
        ):

            search_text = (
                assessment.get("search_text")
                or _build_search_text(
                    assessment
                )
            )

            embedding = self.embedder.encode(
                search_text,
                normalize_embeddings=True,
            ).tolist()

            metadata = {
                "name": assessment.get(
                    "name", ""
                ),

                "url": assessment.get(
                    "url", ""
                ),

                "description": (
                    assessment.get(
                        "description", ""
                    )[:1000]
                ),

                "test_types": ",".join(
                    assessment.get(
                        "test_types", []
                    )
                ),

                "job_levels": ",".join(
                    assessment.get(
                        "job_levels", []
                    )
                ),

                "languages": ",".join(
                    assessment.get(
                        "languages", []
                    )
                ),

                "skills": ",".join(
                    assessment.get(
                        "skills", []
                    )
                ),

                "remote_testing": str(
                    assessment.get(
                        "remote_testing",
                        False,
                    )
                ),

                "adaptive_irt": str(
                    assessment.get(
                        "adaptive_irt",
                        False,
                    )
                ),

                "duration": str(
                    assessment.get(
                        "duration", ""
                    )
                ),
            }

            ids.append(
                f"assessment_{idx}"
            )

            docs.append(search_text)

            embeddings.append(embedding)

            metadatas.append(metadata)

        batch_size = 50

        for start in range(
            0,
            len(ids),
            batch_size,
        ):

            end = start + batch_size

            self.collection.upsert(
                ids=ids[start:end],
                documents=docs[start:end],
                embeddings=embeddings[
                    start:end
                ],
                metadatas=metadatas[
                    start:end
                ],
            )

            logger.info(
                f"Indexed "
                f"{min(end, len(ids))}/"
                f"{len(ids)}"
            )

        logger.info(
            "Chroma index built successfully"
        )

    # ==========================================================
    # RETRIEVE
    # ==========================================================

    def retrieve(
        self,
        query: str,
        n_results: int = 20,
        filter_remote: Optional[
            bool
        ] = None,
        filter_types: Optional[
            list[str]
        ] = None,
    ) -> list[dict]:

        if self.collection is None:
            raise RuntimeError(
                "Index not built."
            )

        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        where_filter = _build_where_filter(
            filter_remote,
            filter_types,
        )

        try:

            results = self.collection.query(
                query_embeddings=[
                    query_embedding
                ],

                n_results=min(
                    max(n_results * 2, 20),
                    self.collection.count(),
                ),

                where=(
                    where_filter
                    if where_filter
                    else None
                ),

                include=[
                    "metadatas",
                    "documents",
                    "distances",
                ],
            )

        except Exception as e:

            logger.warning(
                f"Filtered query failed: {e}"
            )

            results = self.collection.query(
                query_embeddings=[
                    query_embedding
                ],

                n_results=min(
                    max(n_results * 2, 20),
                    self.collection.count(),
                ),

                include=[
                    "metadatas",
                    "documents",
                    "distances",
                ],
            )

        query_tokens = _tokenize(query)

        candidates = []

        for meta, doc, dist in zip(
            results["metadatas"][0],
            results["documents"][0],
            results["distances"][0],
        ):

            semantic_score = (
                1 - dist
                if dist is not None
                else 0.0
            )

            keyword_score = (
                _keyword_overlap_score(
                    query_tokens,
                    doc or "",
                )
                or 0.0
            )

            role_bonus = (
                _role_alignment_bonus(
                    query or "",
                    doc or "",
                )
                or 0.0
            )

            hybrid_score = (
                0.60 * semantic_score
                + 0.25 * keyword_score
                + 0.15 * role_bonus
            )

            candidates.append({
                **meta,

                "search_text": doc,

                "semantic_score": round(
                    semantic_score,
                    4,
                ),

                "keyword_score": round(
                    keyword_score,
                    4,
                ),

                "role_bonus": round(
                    role_bonus,
                    4,
                ),

                "hybrid_score": round(
                    hybrid_score,
                    4,
                ),
            })

        candidates = sorted(
            candidates,
            key=lambda x: x[
                "hybrid_score"
            ],
            reverse=True,
        )

        return candidates[:n_results]

    # ==========================================================
    # EXACT MATCH RETRIEVAL
    # ==========================================================

    def get_by_names(
        self,
        names: list[str],
    ) -> list[dict]:

        if self.collection is None:
            raise RuntimeError(
                "Index not built."
            )

        results = self.collection.get(
            include=[
                "metadatas",
                "documents",
            ]
        )

        target_names = {
            n.lower().strip()
            for n in names
        }

        matched = []

        for meta, doc, dist in zip(
            results.get("metadatas", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("distances", [[]])[0],
        ):

            if (
                meta["name"]
                .lower()
                .strip()
                in target_names
            ):

                matched.append({
                    **meta,
                    "search_text": doc,
                })

        return matched

    # ==========================================================
    # ALL NAMES
    # ==========================================================

    def get_all_names(self) -> list[str]:

        if self.collection is None:
            return []

        results = self.collection.get(
            include=["metadatas"]
        )

        return [
            meta["name"]
            for meta in results[
                "metadatas"
            ]
        ]


# ==========================================================
# HELPERS
# ==========================================================

def _build_where_filter(
    filter_remote: Optional[bool],
    filter_types: Optional[
        list[str]
    ],
) -> Optional[dict]:

    conditions = []

    if filter_remote is True:

        conditions.append({
            "remote_testing": {
                "$eq": "True"
            }
        })

    if filter_types:

        type_conditions = []

        for t in filter_types:

            type_conditions.append({
                "test_types": {
                    "$contains": t
                }
            })

        if len(type_conditions) == 1:
            conditions.append(
                type_conditions[0]
            )

        else:
            conditions.append({
                "$or": type_conditions
            })

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions
    }


def _build_search_text(
    assessment: dict,
) -> str:

    sections = []

    if assessment.get("name"):
        sections.append(
            f"Assessment Name: "
            f"{assessment['name']}"
        )

    if assessment.get("description"):
        sections.append(
            f"Description: "
            f"{assessment['description']}"
        )

    if assessment.get("test_types"):
        sections.append(
            f"Assessment Types: "
            f"{', '.join(assessment['test_types'])}"
        )

    if assessment.get("skills"):
        sections.append(
            f"Skills: "
            f"{', '.join(assessment['skills'])}"
        )

    if assessment.get("job_levels"):
        sections.append(
            f"Job Levels: "
            f"{', '.join(assessment['job_levels'])}"
        )

    if assessment.get("languages"):
        sections.append(
            f"Languages: "
            f"{', '.join(assessment['languages'])}"
        )

    if assessment.get("duration"):
        sections.append(
            f"Duration: "
            f"{assessment['duration']}"
        )

    sections.append(
        f"Remote Testing: "
        f"{'Yes' if assessment.get('remote_testing') else 'No'}"
    )

    sections.append(
        f"Adaptive/IRT: "
        f"{'Yes' if assessment.get('adaptive_irt') else 'No'}"
    )

    return "\n".join(sections)


def _tokenize(
    text: str,
) -> set[str]:

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text,
    )

    stopwords = {
        "the",
        "a",
        "an",
        "for",
        "with",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "need",
        "looking",
        "want",
        "hiring",
        "candidate",
        "role",
    }

    return {
        t
        for t in tokens
        if t not in stopwords
    }


def _keyword_overlap_score(
    query_tokens: set[str],
    document: str,
) -> float:

    if not query_tokens:
        return 0.0

    doc_tokens = _tokenize(document)

    overlap = query_tokens.intersection(
        doc_tokens
    )

    return len(overlap) / len(query_tokens)


def _role_alignment_bonus(
    query: str,
    document: str,
) -> float:

    query_lower = query.lower()
    document_lower = document.lower()

    score = 0.0

    role_patterns = {

        # Graduate / Early Career
        "graduate": 0.15,
        "entry level": 0.15,
        "graduate trainee": 0.15,

        # Leadership
        "leadership": 0.20,
        "executive": 0.20,
        "director": 0.20,
        "cxo": 0.20,
        "benchmark": 0.20,

        # Technical
        "developer": 0.15,
        "engineer": 0.15,
        "java": 0.15,
        "rust": 0.20,
        "linux": 0.15,
        "networking": 0.15,
        "aws": 0.15,
        "docker": 0.15,

        # Contact Center
        "customer service": 0.20,
        "contact center": 0.20,
        "call center": 0.20,
        "spoken english": 0.20,

        # Safety / Industrial
        "safety": 0.20,
        "dependability": 0.20,
        "manufacturing": 0.20,
        "industrial": 0.20,
        "chemical": 0.20,

        # Healthcare
        "healthcare": 0.20,
        "hipaa": 0.20,
        "medical": 0.20,
        "patient": 0.20,

        # Office/Admin
        "excel": 0.20,
        "word": 0.20,
        "administrative": 0.20,
        "admin assistant": 0.20,

        # Finance
        "financial": 0.20,
        "accounting": 0.20,
        "numerical": 0.15,
        "statistics": 0.15,
    }

    for keyword, bonus in role_patterns.items():

        if (
            keyword in query_lower
            and keyword in document_lower
        ):
            score += bonus

        # ======================================================
    # DIRECT ASSESSMENT BOOSTS
    # ======================================================

    assessment_boosts = {

        "opq": [
            "leadership",
            "executive",
            "graduate",
            "sales",
        ],

        "verify": [
            "graduate",
            "analytical",
            "numerical",
            "reasoning",
        ],

        "dsi": [
            "safety",
            "dependability",
            "industrial",
        ],

        "svar": [
            "spoken english",
            "contact center",
            "customer service",
        ],

        "excel": [
            "excel",
            "admin",
        ],

        "word": [
            "word",
            "admin",
        ],
    }

    for assessment_term, triggers in assessment_boosts.items():

        if assessment_term in document_lower:

            for trigger in triggers:

                if trigger in query_lower:
                    score += 0.10