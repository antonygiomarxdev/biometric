"""Seed demo data for E2E testing."""
import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.core.config import config
from src.db.models import Case, Evidence, Decision, AuditLog

engine = create_engine(config.database_url)
now = datetime.now(timezone.utc)

with Session(engine) as session:
    cases = [
        Case(id=uuid.uuid4(), case_number="CASO-2025-001", title="Homicidio en el Distrito I", description="Huella latente en arma blanca.", status="open", created_at=now - timedelta(days=30), updated_at=now),
        Case(id=uuid.uuid4(), case_number="CASO-2025-002", title="Robo a mano armada", description="Huellas del mostrador.", status="open", created_at=now - timedelta(days=25), updated_at=now),
        Case(id=uuid.uuid4(), case_number="CASO-2025-003", title="Lesiones Personales", description="Huellas parciales en arma.", status="open", created_at=now - timedelta(days=20), updated_at=now),
    ]
    session.add_all(cases)
    session.flush()

    evs = []
    for c in cases:
        for i in range(2):
            evs.append(Evidence(id=uuid.uuid4(), case_id=c.id, fingerprint_id=f"FP-{i+1}", image_path=f"evidences/{c.id}/latent_{i+1}.png", num_minutiae=45 + i*10, created_at=now - timedelta(days=15), updated_at=now))
    session.add_all(evs)
    session.flush()

    session.add(Decision(id=uuid.uuid4(), case_id=cases[0].id, evidence_id=evs[0].id, verdict="Identificación", comments="12 puntos coincidentes.", created_at=now))
    session.add(Decision(id=uuid.uuid4(), case_id=cases[0].id, evidence_id=evs[1].id, verdict="Exclusión", comments="Sin coincidencia.", created_at=now))
    session.flush()

    for c in cases:
        session.add(AuditLog(id=uuid.uuid4(), table_name="cases", record_id=c.id, action="INSERT", payload={"case_number": c.case_number}, previous_hash=None, current_hash="0"*64, created_at=now))

    session.commit()
    print(f"✅ Seeded: {len(cases)} cases, {len(evs)} evidence, decisions, logs.")
