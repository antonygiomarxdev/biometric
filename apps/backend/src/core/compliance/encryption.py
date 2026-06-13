"""Symmetric encryption service using Fernet (AES-256-CBC + HMAC-SHA256).

Provides client-side encryption for object storage blob data. The
EncryptionService is injected into storage adapters via dependency
injection — the storage layer never imports compliance strategy code
directly.
"""

from __future__ import annotations

import logging
from typing import Final

from cryptography.fernet import Fernet, InvalidToken

from src.core.config import config

logger = logging.getLogger(__name__)


class EncryptionService:
    """Encrypt and decrypt bytes using Fernet symmetric encryption.

    Uses AES-256 in CBC mode with HMAC-SHA256 authentication via the
    ``cryptography.fernet`` implementation.  The service is keyed from
    ``config.storage_encryption_key`` when no explicit key is provided.

    Args:
        key: A base64-encoded 32-byte Fernet key.  When ``None``, the
            key is read from ``config.storage_encryption_key``.

    Raises:
        ValueError: If no key is available (both argument and config
            are empty).

    Example:
        >>> service = EncryptionService()
        >>> ciphertext = service.encrypt(b"plaintext")
        >>> plaintext = service.decrypt(ciphertext)
        >>> plaintext == b"plaintext"
        True
    """

    __slots__ = ("_fernet",)

    def __init__(self, key: str | None = None) -> None:
        resolved_key: str = key if key else config.storage_encryption_key
        if not resolved_key:
            raise ValueError(
                "No encryption key provided.  Set the STORAGE_ENCRYPTION_KEY "
                "environment variable or pass a key to the constructor."
            )
        self._fernet: Final[Fernet] = Fernet(resolved_key.encode("ascii"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt *data* using the configured Fernet key.

        Args:
            data: Plaintext bytes to encrypt (may be empty).

        Returns:
            Fernet-encrypted ciphertext bytes.
        """
        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt *data* that was previously encrypted with this service.

        Args:
            data: Fernet ciphertext bytes.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            InvalidToken: If *data* is not valid Fernet ciphertext or
                was encrypted with a different key.
        """
        return self._fernet.decrypt(data)
