import { Star } from "lucide-react";
import type { MatchCandidate } from "@/lib/api";

interface CandidateCardProps {
  candidate: MatchCandidate;
  rank: number;
  isSelected: boolean;
  onSelect: () => void;
}

function scoreColorClass(score: number): string {
  if (score >= 0.8) return "bg-green-500";
  if (score >= 0.5) return "bg-yellow-500";
  return "bg-muted-foreground/30";
}

export function CandidateCard({
  candidate,
  rank,
  isSelected,
  onSelect,
}: CandidateCardProps) {
  const scorePercent = ((candidate.score ?? 0) * 100).toFixed(1);
  const traceCount = candidate.supporting_pairs.length;

  return (
    <div
      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
        isSelected
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/50 hover:bg-muted/30"
      }`}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      tabIndex={0}
      role="button"
      aria-selected={isSelected}
      aria-label={`Candidato ${rank}: ${candidate.full_name ?? "Sin nombre"}`}
    >
      <div className="flex items-start gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary text-sm font-bold shrink-0">
          {rank === 1 ? (
            <Star className="w-4 h-4 fill-primary text-primary" />
          ) : (
            rank
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="font-medium text-sm truncate">
            {candidate.full_name ?? `Persona ${candidate.person_id.slice(0, 8)}`}
          </p>
          <p className="text-xs text-muted-foreground font-mono truncate">
            {candidate.external_id ?? candidate.person_id}
          </p>
          <div className="flex items-center gap-2 mt-1.5">
            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${scoreColorClass(candidate.score ?? 0)}`}
                style={{ width: `${Math.min(Number(scorePercent), 100)}%` }}
              />
            </div>
            <span className="text-xs font-mono text-muted-foreground shrink-0">
              {scorePercent}%
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            {candidate.peak_votes} pares coincidentes
            {traceCount > 0 && (
              <>
                <span className="text-muted-foreground/60"> · </span>
                <span className="text-primary">{traceCount} supporting</span>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
