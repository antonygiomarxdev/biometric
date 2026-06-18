"""Servicio para interacción con Object Storage (MinIO/S3)."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from cryptography.fernet import InvalidToken
from minio import Minio
from minio.error import S3Error

from src.core.config import config

if TYPE_CHECKING:
    from src.core.compliance.encryption import EncryptionService
    from src.core.compliance.strategy import IComplianceStrategy

logger = logging.getLogger(__name__)


class ObjectStorage:
    """Cliente wrapper para MinIO con soporte opcional de cifrado.

    Storage layer never imports compliance strategy concrete classes
    directly — it receives an ``IComplianceStrategy`` and an
    ``EncryptionService`` via constructor injection.
    """

    client: Minio | None

    def __init__(
        self,
        strategy: IComplianceStrategy | None = None,
        encryption_service: EncryptionService | None = None,
    ) -> None:
        """Inicializa el cliente de MinIO.

        Args:
            strategy: Optional compliance strategy that dictates
                whether client-side encryption must be applied before
                upload.  When ``None``, no encryption is performed.
            encryption_service: Optional service used to encrypt uploads
                and decrypt downloads.  Required when *strategy*
                requires client-side encryption.
        """
        self._strategy: IComplianceStrategy | None = strategy
        self._encryption_service: EncryptionService | None = encryption_service

        try:
            self.client = Minio(
                config.minio_endpoint,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key,
                secure=config.minio_secure,
            )
            self.bucket = config.minio_bucket
            self._ensure_bucket_exists()
        except Exception:
            logger.exception("Error inicializando MinIO")
            self.client = None

    # ------------------------------------------------------------------
    # Public configuration (post-init DI)
    # ------------------------------------------------------------------

    def configure_encryption(
        self,
        strategy: IComplianceStrategy | None,
        encryption_service: EncryptionService | None,
    ) -> None:
        """Configure or reconfigure encryption after construction.

        This method exists to support the global ``storage`` singleton
        pattern.  New code should prefer constructor injection.
        """
        self._strategy = strategy
        self._encryption_service = encryption_service

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_bucket_exists(self) -> None:
        """Verifica que el bucket exista, si no, intenta crearlo."""
        if not self.client:
            return

        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Bucket '%s' creado exitosamente.", self.bucket)
        except Exception:
            logger.exception("Error verificando bucket '%s'", self.bucket)

    def _should_encrypt(self) -> bool:
        """Return True when the active strategy mandates encryption."""
        return bool(
            self._strategy and self._strategy.requires_client_side_encryption()
        )

    # ------------------------------------------------------------------
    # Public API — upload / download / presigned URL
    # ------------------------------------------------------------------

    def upload_file(
        self,
        file_data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> str | None:
        """Sube un archivo al bucket, cifrándolo si la estrategia lo exige.

        Args:
            file_data: Bytes del archivo.
            object_name: Nombre del objeto (ruta) en el bucket.
            content_type: Tipo MIME del archivo.

        Returns:
            El object_name si fue exitoso, None si falló.
        """
        if not self.client:
            logger.warning("Cliente MinIO no disponible. No se subió el archivo.")
            return None

        # Apply client-side encryption when the compliance strategy mandates it
        data_to_store: bytes = file_data
        if self._should_encrypt() and self._encryption_service is not None:
            logger.info(
                "Cifrando %s antes de subir (estrategia: %s)",
                object_name,
                type(self._strategy).__name__ if self._strategy else "?",
            )
            data_to_store = self._encryption_service.encrypt(file_data)

        try:
            data_stream = io.BytesIO(data_to_store)
            self.client.put_object(
                self.bucket,
                object_name,
                data_stream,
                length=len(data_to_store),
                content_type=content_type,
            )
            logger.info("Archivo subido: %s", object_name)
        except S3Error:
            logger.exception("Error subiendo archivo a MinIO")
            return None
        else:
            return object_name

    def download_file(self, object_name: str) -> bytes | None:
        """Descarga un archivo del bucket y lo descifra si es necesario.

        Transparently handles backward compatibility with files that
        were uploaded *before* encryption was enabled: if decryption
        fails with ``InvalidToken``, the raw bytes are returned.

        Args:
            object_name: Nombre del objeto a descargar.

        Returns:
            Bytes del archivo (descifrados si corresponde) o
            None si falla.
        """
        raw_data: bytes | None = self.get_file(object_name)
        if raw_data is None or self._encryption_service is None:
            return raw_data

        # Attempt decryption — transparently fall back for legacy files
        try:
            return self._encryption_service.decrypt(raw_data)
        except InvalidToken:
            logger.info(
                "El objeto %s no está cifrado o usó una clave diferente "
                "— devolviendo bytes sin descifrar.",
                object_name,
            )
            return raw_data

    def get_file(self, object_name: str) -> bytes | None:
        """Descarga un archivo del bucket (sin descifrado).

        This is a low-level method that bypasses any encryption layer.
        Most callers should prefer ``download_file`` to get automatic
        decryption.

        Args:
            object_name: Nombre del objeto a descargar.

        Returns:
            Bytes del archivo o None si falla.
        """
        if not self.client:
            return None

        response = None
        try:
            response = self.client.get_object(self.bucket, object_name)
            return response.read()
        except Exception:
            logger.exception("Error descargando archivo '%s'", object_name)
            return None
        finally:
            if response is not None:
                response.close()
                if hasattr(response, "release_conn"):
                    response.release_conn()

    def get_presigned_url(self, object_name: str) -> str | None:
        """Genera una URL firmada temporal para acceso directo.

        Args:
            object_name: Nombre del objeto.

        Returns:
            URL firmada o None.
        """
        if not self.client:
            return None

        try:
            return self.client.get_presigned_url("GET", self.bucket, object_name)
        except Exception:
            logger.exception("Error generando presigned URL")
            return None


# Instancia global (sin cifrado por defecto — backward compatible)
storage = ObjectStorage()
