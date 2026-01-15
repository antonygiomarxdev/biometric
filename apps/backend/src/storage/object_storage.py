"""Servicio para interacción con Object Storage (MinIO/S3)."""

import logging
import io
from typing import Optional, BinaryIO
from minio import Minio
from minio.error import S3Error
from src.core.config import config

logger = logging.getLogger(__name__)

class ObjectStorage:
    """Cliente wrapper para MinIO."""
    
    def __init__(self):
        """Inicializa el cliente de MinIO."""
        try:
            self.client = Minio(
                config.minio_endpoint,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key,
                secure=config.minio_secure
            )
            self.bucket = config.minio_bucket
            self._ensure_bucket_exists()
        except Exception as e:
            logger.error(f"Error inicializando MinIO: {e}")
            self.client = None

    def _ensure_bucket_exists(self):
        """Verifica que el bucket exista, si no, intenta crearlo."""
        if not self.client:
            return

        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Bucket '{self.bucket}' creado exitosamente.")
        except Exception as e:
            logger.error(f"Error verificando bucket '{self.bucket}': {e}")

    def upload_file(self, file_data: bytes, object_name: str, content_type: str = "application/octet-stream") -> Optional[str]:
        """Sube un archivo al bucket.
        
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

        try:
            data_stream = io.BytesIO(file_data)
            self.client.put_object(
                self.bucket,
                object_name,
                data_stream,
                length=len(file_data),
                content_type=content_type
            )
            logger.info(f"Archivo subido: {object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"Error subiendo archivo a MinIO: {e}")
            return None

    def get_file(self, object_name: str) -> Optional[bytes]:
        """Descarga un archivo del bucket.
        
        Args:
            object_name: Nombre del objeto a descargar.
            
        Returns:
            Bytes del archivo o None si falla.
        """
        if not self.client:
            return None

        try:
            response = self.client.get_object(self.bucket, object_name)
            return response.read()
        except Exception as e:
            logger.error(f"Error descargando archivo '{object_name}': {e}")
            return None
        finally:
            if 'response' in locals():
                response.close()
                
            if 'response' in locals() and hasattr(response, 'release_conn'):
                response.release_conn()

    def get_presigned_url(self, object_name: str) -> Optional[str]:
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
        except Exception as e:
            logger.error(f"Error generando presigned URL: {e}")
            return None

# Instancia global
storage = ObjectStorage()
