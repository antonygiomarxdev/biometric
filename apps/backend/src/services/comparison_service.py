"""Fingerprint comparison and matching service."""

from typing import Optional
import logging

from src.core.types import NormalizedFingerprint, MatchResult, Fingerprint
from src.storage.repository import repository
from src.core.config import config
from src.core.metrics import measure_time
from src.storage.object_storage import storage
from typing import Optional, List, Dict
import json

logger = logging.getLogger(__name__)


class ComparisonService:
    """Service for fingerprint comparison and identification."""
    
    def __init__(self, repository=None):
        """
        Args:
            repository: Fingerprint repository (uses global if None)
        """
        from src.storage.repository import repository as default_repo
        self.repository = repository or default_repo
    
    def register_fingerprint(
        self,
        fingerprint: NormalizedFingerprint,
        person_id: str,
        name: str,
        document: str,
        image_bytes: Optional[bytes] = None
    ) -> int:
        """Register a new fingerprint in the system.
        
        Args:
            fingerprint: Processed fingerprint (NormalizedFingerprint)
            person_id: Person ID
            name: Full name
            document: Document number
            image_bytes: Original image bytes (optional, to store in Object Storage)
            
        Returns:
            Database record ID
        """
        if not fingerprint.minutiae:
            raise ValueError("La huella no tiene minutiae extraídas")
        
        # 1. Upload image to Object Storage (if provided)
        image_path = None
        if image_bytes:
            try:
                # We use person_id as part of the filename for organization
                # In a real system we would use a unique UUID to avoid collisions if a person has multiple fingerprints
                object_name = f"raw/{person_id}_{document}.bmp"
                image_path = storage.upload_file(image_bytes, object_name, content_type="image/bmp")
            except Exception as e:
                logger.error(f"Error subiendo imagen para {person_id}: {e}")
                # We don't block registration if storage fails, but we log the error
        
        # 2. Prepare minutiae data for reproducibility
        minutiae_data = [
            {
                "x": m.x,
                "y": m.y,
                "type": m.type.value,  # MinutiaType is an Enum, use .value to get the integer
                "angle": m.angle,
                "confidence": m.confidence
            } for m in fingerprint.minutiae
        ]
        
        # 3. Register in DB
        record_id = self.repository.register(
            fp=fingerprint,
            person_id=person_id,
            name=name,
            doc=document,
            image_path=image_path,
            minutiae_data=minutiae_data
        )
        
        logger.info(
            f"Huella registrada: person_id={person_id}, "
            f"minutiae={len(fingerprint.minutiae)}, db_id={record_id}, image={image_path}"
        )
        
        return record_id
    
    def identify(self, fingerprint: NormalizedFingerprint) -> MatchResult:
        """Identify a fingerprint in the system.
        
        Args:
            fingerprint: Fingerprint to identify (NormalizedFingerprint)
            
        Returns:
            MatchResult with comparison result
        """
        # The repository now has identify() (I'll add it soon)
        # Or we could use match() async, but this service seems synchronous.
        # The user asked "Add synchronous repository.identify()".
        # We assume repository.identify will exist and accept NormalizedFingerprint.
        
        return self.repository.identify(
            fingerprint,
            top_k=config.top_k_matches
        )


# Global instance of the comparison service
comparison_service = ComparisonService()
