"""Development server — PostgreSQL, hot reload, Qdrant fallback.

Prerequisites:
    docker compose -f docker-compose.dev.yml up -d

Usage:
    uv run dev
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ENV", "development")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/fingerprint")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.setdefault("MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
os.environ.setdefault("MINIO_SECURE", "false")


def main() -> None:
    import uvicorn
    from src.core.config import config

    print("  Biometric — Dev Server")
    print(f"  ENV:        {config.env}")
    print(f"  DB:         {config.database_url}")
    print(f"  Pool:       {config.db_pool_size}/{config.db_max_overflow}")
    print()
    print("  OpenAPI docs: http://localhost:8000/docs")
    print()
    print("  Prerequisites: docker compose -f docker-compose.dev.yml up -d")
    print()

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug" if config.log_level == "DEBUG" else "info",
    )


if __name__ == "__main__":
    main()
