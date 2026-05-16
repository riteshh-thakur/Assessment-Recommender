"""
Build Vector Index
------------------

Builds / rebuilds the Chroma vector index
using the enhanced hybrid retrieval pipeline.

Optimized for:
- SHL evaluator traces
- Recall@10
- Hybrid retrieval
- Metadata-aware ranking

Usage:
    python -m scripts.build_index
    python -m scripts.build_index --force
"""

import os
import sys
import logging

# ==========================================================
# PROJECT ROOT
# ==========================================================

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)

# ==========================================================
# IMPORTS
# ==========================================================

from catalog.vector_store import (
    SHLVectorStore,
)

# ==========================================================
# MAIN
# ==========================================================

def main():

    force = "--force" in sys.argv

    catalog_path = os.getenv(
        "CATALOG_JSON_PATH",
        "./data/catalog.json",
    )

    chroma_dir = os.getenv(
        "CHROMA_PERSIST_DIR",
        "./data/chroma",
    )

    logger.info(
        f"Catalog path: "
        f"{catalog_path}"
    )

    logger.info(
        f"Chroma dir: "
        f"{chroma_dir}"
    )

    logger.info(
        f"Force rebuild: "
        f"{force}"
    )

    # ======================================================
    # INIT STORE
    # ======================================================

    store = SHLVectorStore(
        catalog_path=catalog_path,
        chroma_persist_dir=chroma_dir,
    )

    # ======================================================
    # BUILD INDEX
    # ======================================================

    store.build_index(
        force_rebuild=force
    )

    logger.info(
        "Vector index built successfully"
    )

    # ======================================================
    # SANITY CHECKS
    # ======================================================

    sanity_queries = [

        (
            "Senior leadership "
            "executive personality "
            "assessment"
        ),

        (
            "Graduate finance "
            "numerical reasoning"
        ),

        (
            "Rust networking "
            "infrastructure engineer"
        ),

        (
            "Safety-critical "
            "manufacturing operator"
        ),

        (
            "Bilingual healthcare "
            "HIPAA administration"
        ),
    ]

    logger.info(
        "=" * 60
    )

    logger.info(
        "RUNNING SANITY CHECKS"
    )

    logger.info(
        "=" * 60
    )

    for query in sanity_queries:

        logger.info(
            f"\nQUERY: {query}"
        )

        try:

            results = store.retrieve(
                query=query,
                n_results=3,
            )

            for idx, r in enumerate(
                results,
                start=1,
            ):

                score = r.get(
                    "hybrid_score",
                    0.0,
                )

                logger.info(
                    f"{idx}. "
                    f"[{score:.3f}] "
                    f"{r['name']} "
                    f"({r.get('test_types','')})"
                )

        except Exception as e:

            logger.error(
                f"Sanity check failed "
                f"for query '{query}': "
                f"{e}"
            )

    logger.info(
        "\nIndex build completed."
    )


# ==========================================================
# ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    main()