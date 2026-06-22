# ⚠️ SUPERSEDED — DO NOT IMPLEMENT

This phase directory is **historical research that did not ship**.

**Status (2026-06-22):** Replaced by Phase 29 (Deep Embedding / AFR-Net).
The code described here (triplet-based matching, 6-D descriptors,
growing algorithm) was tried, evaluated, and deleted.

**Why it's still here:** The acceptance gate **failed** — 0/5 on
50% center crop, 0/5 on 25% corner crop. Self-match was 5/5 but
real latent matching was unusable. So future contributors don't
re-invent the same approach and discover the same crop failure.

**See:**
- `.planning/STATE.md` §"Phases 24-27 — Classical AFIS Research"
- `docs/LESSONS_LEARNED.md` §"Anti-Patterns Observed"
- `.planning/phases/29-deep-embedding/29-SUMMARY.md` — what actually shipped
