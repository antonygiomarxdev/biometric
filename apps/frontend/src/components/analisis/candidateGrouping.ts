/**
 * Pure helpers for the ResultsPanel candidate list.
 *
 * Extracted from ResultsPanel.tsx so they can be unit-tested without
 * React.  No DOM, no state, no I/O.  Only data shape transformations
 * and label formatting.
 */
import type { MatchCandidate } from "@/lib/api";

const MATCH_THRESHOLD_GOOD = 0.9;
const MATCH_THRESHOLD_FAIR = 0.7;
const SUPPORTING_FINGER_THRESHOLD = 0.5;

/** One logical "subject" entry in the candidate list.  A subject is
 *  the unique ``person_id``; the per-finger matches are kept as
 *  ``matches`` so the perito can see whether the match was driven
 *  by a single finger or by multiple fingers of the same person. */
export interface PersonGroup {
  person_id: string;
  label: string;
  external_id: string | null;
  best_score: number;
  best_match: MatchCandidate;
  matches: MatchCandidate[];
  /** Number of fingers of this person whose match score is at or
   *  above ``SUPPORTING_FINGER_THRESHOLD`` (50%).  Used to render
   *  the multi-finger evidence badge.  Always >= 1 because the
   *  best match is always above threshold (otherwise the candidate
   *  would not be in the response). */
  fingers_above_threshold: number;
}

export interface ScoreTier {
  text: string;
  label: string;
  color: string;
}

/** Group candidates by ``person_id`` and rank each group by its
 *  best finger score.  Within a group, matches are sorted by score
 *  desc so the top-1 finger is always first. */
export function groupByPerson(
  candidates: MatchCandidate[],
): PersonGroup[] {
  const byId = new Map<string, MatchCandidate[]>();
  for (const c of candidates) {
    const arr = byId.get(c.person_id) ?? [];
    arr.push(c);
    byId.set(c.person_id, arr);
  }
  const groups: PersonGroup[] = [];
  for (const [personId, matches] of byId) {
    const sorted = [...matches].sort(
      (a, b) => (b.score ?? 0) - (a.score ?? 0),
    );
    const best = sorted[0];
    const fingers_above_threshold = sorted.filter(
      (m) => (m.score ?? 0) >= SUPPORTING_FINGER_THRESHOLD,
    ).length;
    groups.push({
      person_id: personId,
      label: best.full_name ?? best.external_id ?? personId.slice(0, 8),
      external_id: best.external_id,
      best_score: best.score ?? 0,
      best_match: best,
      matches: sorted,
      fingers_above_threshold,
    });
  }
  groups.sort((a, b) => b.best_score - a.best_score);
  return groups;
}

/** Visual tier for a score in [0, 1].  Used both for the person
 *  row's top-1 score and for each supporting finger's score. */
export function scoreTier(score: number): ScoreTier {
  if (score >= MATCH_THRESHOLD_GOOD) {
    return {
      text: "text-green-500",
      label: "Coincidencia alta",
      color: "#22c55e",
    };
  }
  if (score >= MATCH_THRESHOLD_FAIR) {
    return {
      text: "text-yellow-500",
      label: "Coincidencia media",
      color: "#eab308",
    };
  }
  return {
    text: "text-red-500",
    label: "Coincidencia baja",
    color: "#ef4444",
  };
}

/** Tier for the multi-finger evidence badge.  ``count`` is the
 *  number of fingers of the same person whose match score is at or
 *  above the supporting-finger threshold (50 %).  The badge is only
 *  shown when ``count >= 2``; a single-finger match is just a match,
 *  not "evidence".  We do NOT show a denominator because the
 *  "matches returned in top-K" denominator is confusing (it sounds
 *  like "X of Y enrolled fingers", but Y is actually Y returned,
 *  not Y enrolled). */
export function fingerEvidenceTier(
  count: number,
): {
  label: string;
  className: string;
  tooltip: string;
} | null {
  if (count < 2) return null;
  if (count >= 7) {
    return {
      label: `${count} dedos`,
      className: "bg-green-600/20 text-green-700 border border-green-600/30",
      tooltip: `${count} dedos de esta persona muestran similitud ≥ 50%. Evidencia muy fuerte de identificación correcta.`,
    };
  }
  if (count >= 4) {
    return {
      label: `${count} dedos`,
      className: "bg-green-500/15 text-green-600 border border-green-500/30",
      tooltip: `${count} dedos de esta persona muestran similitud ≥ 50%. Evidencia fuerte.`,
    };
  }
  return {
    label: `${count} dedos`,
    className: "bg-blue-500/15 text-blue-600 border border-blue-500/30",
    tooltip: `${count} dedos de esta persona muestran similitud ≥ 50%. Evidencia moderada.`,
  };
}

/** Compact label for the finger position.  The backend stores
 *  values like "Left index" or "Right thumb"; we render them as
 *  "L · index" / "R · thumb" to fit in a chip.  Falls back to
 *  the raw string when the value is not in the dictionary. */
export function fingerShortLabel(fingerName: string): string {
  const normalized = fingerName.toLowerCase();
  const hand = normalized.includes("left")
    ? "L"
    : normalized.includes("right")
      ? "R"
      : "";
  const position = normalized.includes("thumb")
    ? "Pulgar"
    : normalized.includes("index")
      ? "Índice"
      : normalized.includes("middle")
        ? "Medio"
        : normalized.includes("ring")
          ? "Anular"
          : normalized.includes("little") || normalized.includes("pinky")
            ? "Meñique"
            : "";
  if (hand && position) return `${hand} · ${position}`;
  if (position) return position;
  return fingerName;
}
