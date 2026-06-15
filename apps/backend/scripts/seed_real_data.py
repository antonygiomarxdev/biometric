"""
Seed script: Poblates la BD con datos forenses realistas para pruebas E2E.
Ejecutar: cd apps/backend && PYTHONPATH=. python scripts/seed_real_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.core.config import config
from src.db.models import Base, Case, Evidence, Decision, User, AuditLog

def seed():
    engine = create_engine(config.database_url)
    Base.metadata.create_all(bind=engine)
    
    with Session(engine) as session:
        # --- Users ---
        admin = User(
            id=uuid.uuid4(),
            username="admin",
            email="admin@forenso.local",
            hashed_password="$2b$12$NwUqRJmS2J3Y6I7K8L9M0O1P2Q3R4S5T6U7V8W9X0Y1Z2a3b4c5d6e",
            role="Admin",
            full_name="Administrador del Sistema",
            is_active=True,
        )
        perito = User(
            id=uuid.uuid4(),
            username="perito1",
            email="perito1@forenso.local",
            hashed_password="$2b$12$NwUqRJmS2J3Y6I7K8L9M0O1P2Q3R4S5T6U7V8W9X0Y1Z2a3b4c5d6f",
            role="Perito",
            full_name="Perito Forense Uno",
            is_active=True,
        )
        session.add_all([admin, perito])
        session.flush()
        print(f"✅ Users created: admin, perito1")

        # --- Cases ---
        now = datetime.now(timezone.utc)
        cases_data = [
            ("CASO-2024-001", "Homicidio en el Distrito I", "Homicidio",
             "Hallazgo de huella latente en arma blanca encontrada en la escena del crimen."),
            ("CASO-2024-002", "Robo a mano armada — Farmacia Central", "Robo",
             "Huellas levantadas del mostrador y la caja registradora."),
            ("CASO-2024-003", "Violación en la Colonia Bosques", "Violación",
             "Huellas digitales y palmares levantados de la escena."),
            ("CASO-2024-004", "Lesiones personales — Riña Campal", "Lesiones",
             "Evidencia de arma de fuego con huellas parciales."),
        ]
        cases = []
        for case_num, title, crime_type, desc in cases_data:
            c = Case(
                id=uuid.uuid4(),
                case_number=case_num,
                title=title,
                description=desc,
                status="open",
                crime_type=crime_type,
                created_at=now - timedelta(days=30),
                updated_at=now,
            )
            cases.append(c)
        session.add_all(cases)
        session.flush()
        print(f"✅ {len(cases)} cases created")

        # --- Evidences (metadata only, no real image files) ---
        evidences = []
        for case in cases:
            for i in range(2):
                e = Evidence(
                    id=uuid.uuid4(),
                    case_id=case.id,
                    fingerprint_id=f"FP-{case.case_number}-{i+1}",
                    image_path=f"evidences/{case.id}/latent_{i+1}.png",
                    num_minutiae=45 + i * 10,
                    created_at=now - timedelta(days=25),
                    updated_at=now,
                )
                evidences.append(e)
        session.add_all(evidences)
        session.flush()
        print(f"✅ {len(evidences)} evidences created (metadata)")

        # --- Decisions (for first case) ---
        for case in cases[:1]:
            d = Decision(
                id=uuid.uuid4(),
                case_id=case.id,
                evidence_id=evidences[0].id if evidences else None,
                verdict="Identificación",
                confidence=0.95,
                comments="Coincidencia positiva: las crestas papilares coinciden en 12 puntos característicos.",
                created_at=now,
                updated_at=now,
            )
            session.add(d)
        session.flush()
        print(f"✅ Decisions created")

        # --- Audit log entries ---
        for case in cases:
            log = AuditLog(
                id=uuid.uuid4(),
                table_name="cases",
                record_id=case.id,
                action="INSERT",
                payload={"case_number": case.case_number, "title": case.title},
                previous_hash=None,
                current_hash="0" * 64,
                created_at=now,
            )
            session.add(log)
        session.flush()
        print(f"✅ {len(cases)} audit log entries created")

        session.commit()
        print("\n✅ Seed data committed successfully!")

if __name__ == "__main__":
    seed()
