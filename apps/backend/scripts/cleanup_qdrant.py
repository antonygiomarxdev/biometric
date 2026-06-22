#!/usr/bin/env python3
"""Clean up legacy Qdrant collections from the MCC/Bozorth3 era.

Phase 29 — removes the following Qdrant collections:
- ``ridge_graphs`` — serialised MCC graph vectors
- ``pair_features`` — L2-normalised 5-D pair vectors
- ``deepprint_poc`` — Phase 24 DeepPrint POC vectors

Run AFTER migration 0010 and BEFORE re-enrolling with the new pipeline.

Usage:
    uv run python scripts/cleanup_qdrant.py
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cleanup_qdrant")

LEGACY_COLLECTIONS = [
    "ridge_graphs",
    "pair_features",
    "deepprint_poc",
]


def main() -> None:
    client = QdrantClient(host="localhost", port=6333)
    existing = {c.name for c in client.get_collections().collections}
    logger.info("Existing collections: %s", existing)

    for name in LEGACY_COLLECTIONS:
        if name in existing:
            client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        else:
            logger.info("Collection '%s' not found — skipping", name)

    remaining = {c.name for c in client.get_collections().collections}
    logger.info("Remaining collections: %s", remaining)


if __name__ == "__main__":
    main()
