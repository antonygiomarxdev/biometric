"""Tests for EncryptionService — symmetric encrypt/decrypt with Fernet."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet, InvalidToken

from src.core.compliance.encryption import EncryptionService


class TestEncryptionService:
    """EncryptionService should encrypt and decrypt bytes symmetrically."""

    @pytest.fixture
    def valid_key(self) -> str:
        """Return a valid Fernet key as a string."""
        return Fernet.generate_key().decode()

    @pytest.fixture
    def service(self, valid_key: str) -> EncryptionService:
        """Return an EncryptionService with a valid key."""
        return EncryptionService(key=valid_key)

    def test_encrypt_decrypt_roundtrip(
        self, service: EncryptionService
    ) -> None:
        """Encrypting then decrypting should return the original bytes."""
        original: bytes = b"Fingerprint image data for testing purposes."
        encrypted: bytes = service.encrypt(original)
        assert encrypted != original, "Ciphertext must differ from plaintext"
        decrypted: bytes = service.decrypt(encrypted)
        assert decrypted == original, "Roundtrip must restore original data"

    def test_encrypt_empty_bytes(self, service: EncryptionService) -> None:
        """Encrypting empty bytes should produce valid ciphertext."""
        encrypted: bytes = service.encrypt(b"")
        assert isinstance(encrypted, bytes)
        decrypted: bytes = service.decrypt(encrypted)
        assert decrypted == b""

    def test_decrypt_invalid_data_raises(
        self, service: EncryptionService
    ) -> None:
        """Decrypting non-Fernet bytes should raise InvalidToken."""
        with pytest.raises(InvalidToken):
            service.decrypt(b"not-fernct-ciphertext")

    def test_instantiation_with_invalid_key_raises(self) -> None:
        """Creating EncryptionService with an invalid key should fail."""
        with pytest.raises(Exception):  # ValueError or struct.error
            EncryptionService(key="not-a-valid-fernet-key")

    def test_different_keys_produce_different_ciphertext(self) -> None:
        """Same plaintext encrypted with different keys must differ."""
        svc1 = EncryptionService(key=Fernet.generate_key().decode())
        svc2 = EncryptionService(key=Fernet.generate_key().decode())
        data: bytes = b"Fixed test message"
        c1: bytes = svc1.encrypt(data)
        c2: bytes = svc2.encrypt(data)
        assert c1 != c2, "Different keys must produce different ciphertext"

    def test_instantiation_without_key_raises_value_error(self) -> None:
        """Creating EncryptionService with no key should raise ValueError."""
        with pytest.raises(ValueError, match="No encryption key provided"):
            EncryptionService()  # type: ignore[call-overload]
