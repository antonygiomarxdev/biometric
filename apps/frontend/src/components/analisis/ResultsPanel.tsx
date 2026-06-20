import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CandidateDetailPanel } from "@/components/fingerprint/CandidateDetailPanel";
import { Trophy, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MatchCandidate, MatchSearchResponse } from "@/lib/api";

const MATCH_THRESHOLD_GOOD = 0.9;
const MATCH_THRESHOLD_FAIR = 0.7;

interface ResultsPanelProps {
  searchResult: MatchSearchResponse | null;
  probeImageUrl: string | null;
  probeMinutiae: any[];
}

export function ResultsPanel({
  searchResult,
  probeImageUrl,
  probeMinutiae,
}: ResultsPanelProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  const selectedCandidate: MatchCandidate | null =
    searchResult?.candidates[selectedIdx] ?? null;

  if (!searchResult) {
    return null;
  }

  if (searchResult.candidates.length === 0) {
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

  return (
    <div>
      <Card className="border-border/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
            <Trophy className="w-3.5 h-3.5" />
            Top {searchResult.candidates.length} — click para comparar
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ul className="divide-y divide-border/40 max-h-[480px] overflow-y-auto">
            {searchResult.candidates.map((c, i) => {
              const isSelected = i === selectedIdx;
              const label =
                c.full_name ?? c.external_id ?? c.person_id.slice(0, 8);
              const score = c.score ?? 0;
              const scoreInfo = {
                text:
                  score >= MATCH_THRESHOLD_GOOD
                    ? "text-green-500"
                    : score >= MATCH_THRESHOLD_FAIR
                    ? "text-yellow-500"
                    : "text-red-500",
                label:
                  score >= MATCH_THRESHOLD_GOOD
                    ? "Coincidencia alta"
                    : score >= MATCH_THRESHOLD_FAIR
                    ? "Coincidencia media"
                    : "Coincidencia baja",
              };
              const barColor = scoreInfo.text.includes("green")
                ? "#22c55e"
                : scoreInfo.text.includes("yellow")
                ? "#eab308"
                : "#ef4444";
              return (
                <li key={c.person_id}>
                  <button
                    onClick={() => setSelectedIdx(i)}
                    className={cn(
                      "w-full text-left px-3 py-2.5 transition-all flex items-center gap-3",
                      "hover:bg-muted/40",
                      isSelected && "bg-primary/10 border-l-2 border-l-primary"
                    )}
                  >
                    <span
                      className={cn(
                        "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold",
                        isSelected
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      )}
                    >
                      {i + 1}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span
                          className={cn(
                            "text-sm truncate",
                            isSelected ? "font-bold" : "font-medium"
                          )}
                        >
                          {label}
                        </span>
                        <span
                          className={cn(
                            "text-[10px] uppercase tracking-wider font-bold",
                            scoreInfo.text
                          )}
                        >
                          {scoreInfo.label}
                        </span>
                        {isSelected && (
                          <span className="ml-auto text-[10px] uppercase tracking-wider font-bold text-primary flex items-center gap-1">
                            <Search className="w-3 h-3" />
                            Comparando
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-[11px] text-muted-foreground mt-0.5 font-mono">
                        <span className={cn("font-bold", scoreInfo.text)}>
                          {Math.round((c.score ?? 0) * 100)}%
                        </span>
                        <span>{c.peak_votes} pares</span>
                        <span>{c.supporting_pairs.length} match</span>
                      </div>
                      <div className="mt-1.5 h-1 bg-muted/60 rounded overflow-hidden">
                        <div
                          className="h-full transition-all"
                          style={{
                            width: `${Math.min((c.score ?? 0) * 100, 100)}%`,
                            backgroundColor: barColor,
                          }}
                        />
                      </div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        </CardContent>
      </Card>

      {selectedCandidate && (
        <div className="mt-3">
          <CandidateDetailPanel
            candidate={selectedCandidate}
            probeImageUrl={probeImageUrl ?? ""}
            probeMinutiae={probeMinutiae}
            onDismiss={() => setSelectedIdx(0)}
          />
        </div>
      )}
    </div>
  );
}
