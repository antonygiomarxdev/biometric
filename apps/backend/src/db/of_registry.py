"""Orientation field persistence via PostgreSQL JSONB.

Stores the 16×16 orientation field + coherence matrices keyed by
``fingerprint_id`` for the OF pre-filter (Phase 26, Plan 26-01, T2).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict, cast

from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import FingerprintOFIndex
from src.processing.of_similarity import OFSimilarity, of_pseudo_core


class OFRecord(TypedDict):
    fingerprint_id: str
    of_ori: list[list[float]]
    of_coh: list[list[float]]
    block_size: int
    pseudo_core: tuple[int, int] | None
    enrolled_at: datetime


def _row_to_record(row: FingerprintOFIndex) -> OFRecord:
    pc: tuple[int, int] | None = None
    if row.pseudo_core_row is not None and row.pseudo_core_col is not None:
        pc = (row.pseudo_core_row, row.pseudo_core_col)
    raw_ori: object = row.of_ori
    raw_coh: object = row.of_coh
    return OFRecord(
        fingerprint_id=str(row.fingerprint_id),
        of_ori=cast("list[list[float]]", raw_ori),
        of_coh=cast("list[list[float]]", raw_coh),
        block_size=int(row.block_size),
        pseudo_core=pc,
        enrolled_at=row.enrolled_at,
    )


class OFRegistry:
    """Async persistence gateway for the ``fingerprint_of_index`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, fingerprint_id: str, of: OFSimilarity) -> None:
        import uuid

        fid = uuid.UUID(fingerprint_id) if isinstance(fingerprint_id, str) else fingerprint_id
        pc = of_pseudo_core(of.coh)

        existing = await self._session.get(FingerprintOFIndex, fid)
        if existing is not None:
            existing.of_ori = of.ori.tolist()
            existing.of_coh = of.coh.tolist()
            existing.block_size = of.block_size
            existing.pseudo_core_row = pc[0]
            existing.pseudo_core_col = pc[1]
            existing.enrolled_at = datetime.now(UTC)
        else:
            self._session.add(FingerprintOFIndex(
                fingerprint_id=fid,
                of_ori=of.ori.tolist(),
                of_coh=of.coh.tolist(),
                block_size=of.block_size,
                pseudo_core_row=pc[0],
                pseudo_core_col=pc[1],
                enrolled_at=datetime.now(UTC),
            ))
        await self._session.commit()

    async def get(self, fingerprint_id: str) -> OFRecord | None:
        import uuid

        fid = uuid.UUID(fingerprint_id) if isinstance(fingerprint_id, str) else fingerprint_id
        row = await self._session.get(FingerprintOFIndex, fid)
        if row is None:
            return None
        return _row_to_record(row)

    async def get_many(
        self, fingerprint_ids: list[str],
    ) -> dict[str, OFRecord]:
        if not fingerprint_ids:
            return {}

        import uuid

        fids = [
            uuid.UUID(fid) if isinstance(fid, str) else fid
            for fid in fingerprint_ids
        ]
        stmt = select(FingerprintOFIndex).where(
            FingerprintOFIndex.fingerprint_id.in_(fids),
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())

        records: dict[str, OFRecord] = {}
        for row in rows:
            rec = _row_to_record(row)
            records[rec["fingerprint_id"]] = rec
        return records

    async def get_all(self) -> dict[str, OFRecord]:
        """Fetch all enrolled OF records.

        Returns a dict keyed by ``fingerprint_id`` (str).
        Useful for the OF pre-filter which needs to compare the probe
        against all enrolled fingerprints.
        """
        stmt = select(FingerprintOFIndex)
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        records: dict[str, OFRecord] = {}
        for row in rows:
            rec = _row_to_record(row)
            records[rec["fingerprint_id"]] = rec
        return records

    async def delete(self, fingerprint_id: str) -> None:
        import uuid

        fid = uuid.UUID(fingerprint_id) if isinstance(fingerprint_id, str) else fingerprint_id
        row = await self._session.get(FingerprintOFIndex, fid)
        if row is not None:
            await self._session.delete(row)
            await self._session.commit()
