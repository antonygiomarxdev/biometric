import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CandidateDetailPanel } from "@/components/fingerprint/CandidateDetailPanel";
import { Trophy, Search, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MatchCandidate, MatchSearchResponse } from "@/lib/api";
import {
  groupByPerson,
  scoreTier,
  fingerEvidenceTier,
  fingerShortLabel,
} from "./candidateGrouping";
import type { PersonGroup } from "./candidateGrouping";

interface ResultsPanelProps {
  searchResult: MatchSearchResponse | null;
  probeImageUrl: string | null;
}

export function ResultsPanel({
  searchResult,
  probeImageUrl,
}: ResultsPanelProps) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const groups = useMemo(
    () => (searchResult ? groupByPerson(searchResult.candidates) : []),
    [searchResult],
  );

  const selectedCandidate: MatchCandidate | null = useMemo(() => {
    if (!searchResult) return null;
    if (selectedKey) {
      const hit = searchResult.candidates.find(
        (c) => `${c.capture_id ?? c.person_id}` === selectedKey,
      );
      if (hit) return hit;
    }
    return groups[0]?.best_match ?? null;
  }, [searchResult, groups, selectedKey]);

  if (!searchResult) {
    return null;
  }

  if (groups.length === 0) {
    return (
      <Card className="border-border/60">
        <CardHeader>
          <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
            Resultados
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">No se encontraron candidatos.</p>
        </CardContent>
      </Card>
    );
  }

  const toggleExpand = (personId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(personId)) next.delete(personId);
      else next.add(personId);
      return next;
    });
  };

  return (
    <div>
      <Card className="border-border/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
            <Trophy className="w-3.5 h-3.5" />
            Top {groups.length} personas
            <span className="text-muted-foreground/60 font-mono text-[10px]">
              ({searchResult.candidates.length} dedos)
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ul className="divide-y divide-border/40 max-h-[480px] overflow-y-auto">
            {groups.map((group, groupIdx) => (
              <PersonRow
                key={group.person_id}
                group={group}
                groupIdx={groupIdx}
                selectedKey={selectedKey}
                groupIds={groups.map((g) => g.person_id)}
                isDefaultExpanded={groupIdx === 0}
                expanded={expanded}
                onSelect={(key) => setSelectedKey(key)}
                onToggleExpand={() => toggleExpand(group.person_id)}
              />
            ))}
          </ul>
        </CardContent>
      </Card>

      {selectedCandidate && (
        <div className="mt-3">
          <CandidateDetailPanel
            candidate={selectedCandidate}
            probeImageUrl={probeImageUrl ?? ""}
            onDismiss={() => setSelectedKey(null)}
          />
        </div>
      )}
    </div>
  );
}

function PersonRow({
  group,
  groupIdx,
  selectedKey,
  expanded,
  isDefaultExpanded,
  onSelect,
  onToggleExpand,
}: {
  group: PersonGroup;
  groupIdx: number;
  selectedKey: string | null;
  expanded: Set<string>;
  isDefaultExpanded: boolean;
  onSelect: (key: string) => void;
  onToggleExpand: () => void;
}) {
  const tier = scoreTier(group.best_score);
  const isOpen =
    expanded.has(group.person_id) ||
    (expanded.size === 0 && isDefaultExpanded);
  const hasMultiple = group.matches.length > 1;
  const bestKey = `${group.best_match.capture_id ?? group.best_match.person_id}`;
  const isSelected = selectedKey === bestKey;

  // The evidence badge shows the count of fingers of this person
  // that matched above the 50 % threshold.  We do not show a
  // denominator because "matches returned in top-K" is confusing.
  const badge = fingerEvidenceTier(group.fingers_above_threshold);

  return (
    <li>
      <button
        onClick={() => onSelect(bestKey)}
        className={cn(
          "w-full text-left px-3 py-2.5 transition-all flex items-center gap-3",
          "hover:bg-muted/40",
          isSelected && "bg-primary/10 border-l-2 border-l-primary",
        )}
      >
        <span
          className={cn(
            "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold",
            isSelected
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground",
          )}
        >
          {groupIdx + 1}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 min-w-0 flex-wrap">
            <span
              className={cn(
                "text-sm truncate",
                isSelected ? "font-bold" : "font-medium",
              )}
            >
              {group.label}
            </span>
            {badge && (
              <span
                title={badge.tooltip}
                className={cn(
                  "shrink-0 inline-flex items-center rounded-md px-1.5 py-0.5 text-[10px] font-semibold tracking-wide",
                  badge.className,
                )}
              >
                {badge.label}
              </span>
            )}
            <span
              className={cn(
                "text-[10px] uppercase tracking-wider font-bold shrink-0",
                tier.text,
              )}
            >
              {tier.label}
            </span>
            {isSelected && (
              <span className="text-[10px] uppercase tracking-wider font-bold text-primary flex items-center gap-1 shrink-0 ml-auto">
                <Search className="w-3 h-3" />
                Comparando
              </span>
            )}
          </div>
          {group.external_id && (
            <div className="text-[11px] text-muted-foreground font-mono truncate mt-0.5">
              {group.external_id}
            </div>
          )}
          <div className="mt-1.5 h-1 bg-muted/60 rounded overflow-hidden">
            <div
              className="h-full transition-all"
              style={{
                width: `${Math.min(group.best_score * 100, 100)}%`,
                backgroundColor: tier.color,
              }}
            />
          </div>
        </div>
        {hasMultiple && (
          <span
            role="button"
            tabIndex={0}
            aria-label={isOpen ? "Colapsar dedos" : "Expandir dedos"}
            onClick={(e) => {
              e.stopPropagation();
              onToggleExpand();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                e.stopPropagation();
                onToggleExpand();
              }
            }}
            className="shrink-0 p-1 rounded hover:bg-muted"
          >
            {isOpen ? (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            )}
          </span>
        )}
      </button>
      {isOpen && hasMultiple && (
        <ul className="bg-muted/20 border-l-2 border-l-border ml-10">
          {group.matches.map((m, mIdx) => {
            const mTier = scoreTier(m.score ?? 0);
            const mKey = `${m.capture_id ?? m.person_id}`;
            const mSelected = selectedKey === mKey;
            const delta = group.best_score - (m.score ?? 0);
            return (
              <li key={`${group.person_id}-${mIdx}-${m.capture_id ?? m.person_id}`}>
                <button
                  onClick={() => onSelect(mKey)}
                  className={cn(
                    "w-full text-left px-3 py-1.5 transition-all flex items-center gap-2",
                    "hover:bg-muted/40",
                    mSelected && "bg-primary/10 border-l-2 border-l-primary",
                  )}
                >
                  <span className="flex-1 min-w-0 flex items-center gap-2">
                    <span className="text-xs text-muted-foreground font-mono">
                      {m.finger_name ? fingerShortLabel(m.finger_name) : "—"}
                    </span>
                    <span className="h-0.5 flex-1 bg-muted/60 rounded overflow-hidden">
                      <span
                        className="block h-full"
                        style={{
                          width: `${Math.min((m.score ?? 0) * 100, 100)}%`,
                          backgroundColor: mTier.color,
                        }}
                      />
                    </span>
                    <span
                      className={cn(
                        "text-xs font-mono font-bold shrink-0",
                        mTier.text,
                      )}
                    >
                      {Math.round((m.score ?? 0) * 100)}%
                    </span>
                    {delta > 0 && (
                      <span className="text-[10px] font-mono text-muted-foreground/60 shrink-0">
                        -{Math.round(delta * 100)}%
                      </span>
                    )}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </li>
  );
}
