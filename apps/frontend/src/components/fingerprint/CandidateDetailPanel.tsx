import { useMemo } from "react";
import { X, Search, Fingerprint as FingerprintIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MatchOverlay } from "@/components/fingerprint/MatchOverlay";
import type { MatchCandidate, MinutiaPoint } from "@/lib/api";

interface CandidateDetailPanelProps {
  candidate: MatchCandidate;
  probeImageUrl: string;
  probeMinutiae: MinutiaPoint[];
  candidateImageUrl: string | null;
  onDismiss: () => void;
}

function pickTopFingerprintId(candidate: MatchCandidate): string | null {
  if (candidate.contributing_fingerprints.length === 0) {
    return candidate.match_trace[0]?.candidate_fingerprint_id ?? null;
  }
  const counts = new Map<string, number>();
  for (const e of candidate.match_trace) {
    counts.set(
      e.candidate_fingerprint_id,
      (counts.get(e.candidate_fingerprint_id) ?? 0) + 1,
    );
  }
  let bestId: string | null = null;
  let bestCount = -1;
  for (const id of candidate.contributing_fingerprints) {
    const c = counts.get(id) ?? 0;
    if (c > bestCount) {
      bestCount = c;
      bestId = id;
    }
  }
  return bestId;
}

export function CandidateDetailPanel({
  candidate,
  probeImageUrl,
  probeMinutiae,
  candidateImageUrl,
  onDismiss,
}: CandidateDetailPanelProps): React.JSX.Element {
  const topFingerprintId = useMemo(
    () => pickTopFingerprintId(candidate),
    [candidate],
  );

  const candidateLabel =
    candidate.full_name ??
    candidate.external_id ??
    `Persona ${candidate.person_id.slice(0, 8)}`;

  return (
    <Card className="border-border/60">
      <CardHeader className="flex flex-row items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <CardTitle className="text-base flex items-center gap-2">
            <FingerprintIcon className="w-4 h-4" />
            Detalle de coincidencia — {candidateLabel}
          </CardTitle>
          {topFingerprintId && candidate.contributing_fingerprints.length > 1 && (
            <p className="text-xs text-muted-foreground mt-1">
              Huella que más aportó:{" "}
              <span className="font-mono text-primary">{topFingerprintId}</span>
            </p>
          )}
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onDismiss}
          aria-label="Cerrar detalle de candidato"
        >
          <X className="w-4 h-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-6">
        {candidateImageUrl ? (
          <MatchOverlay
            probeImageUrl={probeImageUrl}
            probeMinutiae={probeMinutiae}
            candidateImageUrl={candidateImageUrl}
            candidateMinutiae={[]}
            matchTrace={candidate.match_trace}
            candidateLabel={candidateLabel}
          />
        ) : (
          <Card className="border-border/40 bg-muted/10">
            <CardContent className="flex flex-col items-center justify-center py-8 text-muted-foreground">
              <Search className="w-10 h-10 mb-3 opacity-30" />
              <p className="text-sm">Sin imagen del candidato</p>
              <p className="text-xs mt-1">
                La huella enrolada no se ha cargado; solo se muestra la traza
                tabular.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Tabular trace */}
        {candidate.match_trace.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <FingerprintIcon className="w-10 h-10 mb-2 opacity-20" />
            <p className="text-sm font-medium">Sin traza de cilindros</p>
            <p className="text-xs mt-1">
              El candidato no aportó cilindros coincidentes.
            </p>
          </div>
        ) : (
          <div>
            <h3 className="text-sm font-semibold tracking-tight mb-2">
              Pares de cilindros coincidentes
            </h3>
            <div className="rounded-lg border border-border overflow-hidden">
              <table className="w-full text-xs">
                <thead className="bg-muted/30 text-muted-foreground uppercase tracking-wider">
                  <tr>
                    <th className="text-left px-3 py-2 font-medium">
                      Cilindro probe
                    </th>
                    <th className="text-left px-3 py-2 font-medium">
                      Cilindro candidato
                    </th>
                    <th className="text-right px-3 py-2 font-medium">
                      Similitud
                    </th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {candidate.match_trace.map((e, idx) => (
                    <tr
                      key={idx}
                      className="border-t border-border/60 hover:bg-muted/10"
                    >
                      <td className="px-3 py-1.5">
                        <Badge variant="outline" className="text-[10px]">
                          #{e.probe_cylinder_index}
                        </Badge>
                      </td>
                      <td className="px-3 py-1.5 truncate max-w-[280px]">
                        {e.candidate_fingerprint_id.slice(0, 8)}…
                        <span className="text-muted-foreground/60">
                          {" "}/ {e.candidate_capture_id.slice(0, 8)}
                        </span>
                      </td>
                      <td className="px-3 py-1.5 text-right">
                        <span
                          className={
                            e.similarity >= 0.8
                              ? "text-green-500"
                              : e.similarity >= 0.5
                                ? "text-yellow-500"
                                : "text-muted-foreground"
                          }
                        >
                          {(e.similarity * 100).toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
