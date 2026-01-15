"""Servicio de comparación y matching de huellas."""

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
    """Servicio para comparación e identificación de huellas."""
    
    def __init__(self, repository=None):
        """
        Args:
            repository: Repositorio de huellas (usa el global si es None)
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
        """Registra una nueva huella en el sistema.
        
        Args:
            fingerprint: Huella procesada (NormalizedFingerprint)
            person_id: ID de la persona
            name: Nombre completo
            document: Número de documento
            image_bytes: Bytes de la imagen original (opcional, para guardar en Object Storage)
            
        Returns:
            ID del registro en base de datos
        """
        if not fingerprint.minutiae:
            raise ValueError("La huella no tiene minutiae extraídas")
        
        # 1. Subir imagen a Object Storage (si se proporciona)
        image_path = None
        if image_bytes:
            try:
                # Usamos el person_id como parte del nombre del archivo para organización
                # En un sistema real usaríamos un UUID único para evitar colisiones si una persona tiene múltiples huellas
                object_name = f"raw/{person_id}_{document}.bmp"
                image_path = storage.upload_file(image_bytes, object_name, content_type="image/bmp")
            except Exception as e:
                logger.error(f"Error subiendo imagen para {person_id}: {e}")
                # No bloqueamos el registro si falla el storage, pero logueamos el error
        
        # 2. Preparar datos de minucias para reproducibilidad
        minutiae_data = [
            {
                "x": m.x,
                "y": m.y,
                "type": m.type.value,  # MinutiaType es un Enum, usar .value para obtener el entero
                "angle": m.angle,
                "confidence": m.confidence
            } for m in fingerprint.minutiae
        ]
        
        # 3. Registrar en BD
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
        """Identifica una huella en el sistema.
        
        Args:
            fingerprint: Huella a identificar (NormalizedFingerprint)
            
        Returns:
            MatchResult con resultado de la comparación
        """
        # El repositorio ahora tiene identify() (lo añadiré pronto)
        # O podemos usar match() asíncrono, pero este servicio parece síncrono.
        # El usuario pidió "Agregar repository.identify() síncrono".
        # Asumiremos que repository.identify existirá y aceptará NormalizedFingerprint.
        
        return self.repository.identify(
            fingerprint,
            top_k=config.top_k_matches
        )


# Instancia global del servicio de comparación
comparison_service = ComparisonService()
