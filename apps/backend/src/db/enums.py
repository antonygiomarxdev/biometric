"""Forensic enums for the 4-level fingerprint data model (Phase 17)."""

from __future__ import annotations

from enum import Enum


class FingerPosition(int, Enum):
    """NIST FGP (Finger Position Code) — ANSI/NIST-ITL 1-2011."""

    UNKNOWN = 0
    RIGHT_THUMB = 1
    RIGHT_INDEX = 2
    RIGHT_MIDDLE = 3
    RIGHT_RING = 4
    RIGHT_LITTLE = 5
    LEFT_THUMB = 6
    LEFT_INDEX = 7
    LEFT_MIDDLE = 8
    LEFT_RING = 9
    LEFT_LITTLE = 10
    RIGHT_PALM = 11
    LEFT_PALM = 12
    RIGHT_PALM_LATERAL = 13
    LEFT_PALM_LATERAL = 14


class CaptureType(str, Enum):
    """How a fingerprint was captured."""

    ROLLED = "rolled"
    PLAIN = "plain"
    SLAP = "slap"
    LATENT = "latent"
    PALM = "palm"
    SEGMENT = "segment"


class DocumentType(str, Enum):
    """Type of identification document for a Person."""

    CEDULA = "cedula"
    DUI = "dui"
    PASSPORT = "passport"
    INTERNAL_ID = "internal_id"
    DRIVER_LICENSE = "driver_license"
    OTHER = "other"
