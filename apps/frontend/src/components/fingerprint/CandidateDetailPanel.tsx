import { useState } from "react";
import { X, Fingerprint as FingerprintIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { MatchCandidate } from "@/lib/api";

interface CandidateDetailPanelProps {
  candidate: MatchCandidate;
  probeImageUrl: string;
  onDismiss: () => void;
}

/**
 * Phase 29: no minutiae, no match trace.  Just the cosine score, the
 * candidate's image (MinIO), and a side-by-side visual comparison.
 */
export function CandidateDetailPanel({
  candidate,
  probeImageUrl,
  onDismiss,
}: CandidateDetailPanelProps): React.JSX.Element {
  const candidateLabel =
    candidate.full_name ??
    candidate.external_id ??
    `Persona ${candidate.person_id.slice(0, 8)}`;

  const [showOverlay, setShowOverlay] = useState(true);

  return (
    <Card className="border-border/60">
      <CardHeader className="flex flex-row items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <CardTitle className="text-base flex items-center gap-2">
            <FingerprintIcon className="w-4 h-4" />
            Detalle de coincidencia — {candidateLabel}
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            {candidate.finger_name && (
              <span className="font-mono mr-2">{candidate.finger_name}</span>
            )}
            Score coseno:{" "}
            <span className="font-mono text-primary">
              {(candidate.score * 100).toFixed(2)}%
            </span>
          </p>
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
      <CardContent>
        <div className="flex items-center gap-2 mb-3">
          <Button
            size="sm"
            variant={showOverlay ? "default" : "outline"}
            onClick={() => setShowOverlay(true)}
          >
            Comparar
          </Button>
          <Button
            size="sm"
            variant={!showOverlay ? "default" : "outline"}
            onClick={() => setShowOverlay(false)}
            disabled={!candidate.image_url}
          >
            Solo candidato
          </Button>
        </div>

        {showOverlay && candidate.image_url ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <figure className="space-y-1">
              <figcaption className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Probe
              </figcaption>
              <div className="aspect-square bg-black rounded overflow-hidden">
                {probeImageUrl ? (
                  <img
                    src={probeImageUrl}
                    alt="Probe"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-xs text-muted-foreground">—</span>
                )}
              </div>
            </figure>
            <figure className="space-y-1">
              <figcaption className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Candidato
              </figcaption>
              <div className="aspect-square bg-black rounded overflow-hidden">
                <img
                  src={candidate.image_url}
                  alt={candidateLabel}
                  className="w-full h-full object-cover"
                />
              </div>
            </figure>
          </div>
        ) : (
          <div className="aspect-square bg-black rounded overflow-hidden max-w-md mx-auto">
            {candidate.image_url ? (
              <img
                src={candidate.image_url}
                alt={candidateLabel}
                className="w-full h-full object-cover"
              />
            ) : (
              <span className="text-xs text-muted-foreground">
                Imagen del candidato no disponible
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
