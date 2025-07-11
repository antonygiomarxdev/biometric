from typing import Dict, Optional

import psycopg2


class FingerprintRepositoryPostgres:
    """Repository mapping vector ids to person metadata."""

    def __init__(self, dsn: str) -> None:
        self.conn = psycopg2.connect(dsn)

    def get_person_by_vector_id(self, vector_id: int) -> Optional[Dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT person_id, name, document FROM fingerprints WHERE id=%s",
                (vector_id,),
            )
            row = cur.fetchone()
        if row:
            return {"person_id": row[0], "name": row[1], "document": row[2]}
        return None

    def register_person(
        self, vector_id: int, person_id: str, name: str, document: str
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fingerprints(id, person_id, name, document)
                VALUES (%s, %s, %s, %s)
                """,
                (vector_id, person_id, name, document),
            )
            self.conn.commit()
