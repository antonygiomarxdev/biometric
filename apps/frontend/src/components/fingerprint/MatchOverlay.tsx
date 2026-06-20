import { useMemo, useRef } from "react";
import { Fingerprint } from "lucide-react";
import { useMatchCanvas } from "@/hooks/useMatchCanvas";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { MinutiaPoint, SupportingPair, MatchTraceEntry } from "@/lib/api";

interface MatchOverlayProps {
  probeImageUrl: string;
  probeMinutiae: MinutiaPoint[];
  candidateImageUrl: string;
  candidateMinutiae: MinutiaPoint[];
  supportingPairs: SupportingPair[];
  candidateLabel: string;
}

function buildTrace(
  probeMinutiae: MinutiaPoint[],
  supportingPairs: SupportingPair[],
): MatchTraceEntry[] {
  return supportingPairs.map((sp) => {
    const probeP = probeMinutiae[sp.probe_mi_idx];
    return {
      probe_mi_idx: sp.probe_mi_idx,
      probe_x: probeP?.x ?? 0,
      probe_y: probeP?.y ?? 0,
      candidate_x: sp.candidate_mi_x,
      candidate_y: sp.candidate_mi_y,
      candidate_angle: sp.candidate_mi_angle,
      similarity: sp.similarity,
    };
  });
}

export function MatchOverlay(props: MatchOverlayProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const matchTrace = useMemo(
    () => buildTrace(props.probeMinutiae, props.supportingPairs),
    [props.probeMinutiae, props.supportingPairs],
  );

  const { probeCanvasRef, candidateCanvasRef, svgRef } = useMatchCanvas({
    probeImageUrl: props.probeImageUrl,
    probeMinutiae: props.probeMinutiae,
    candidateImageUrl: props.candidateImageUrl,
    candidateMinutiae: props.candidateMinutiae,
    matchTrace,
    containerRef,
  });

  const matchedCount = matchTrace.length;
  const avgSimilarity =
    matchedCount > 0
      ? matchTrace.reduce((s, e) => s + e.similarity, 0) / matchedCount
      : 0;
  const probeMinCount = props.probeMinutiae.length;

  return (
    <Card className="border-border/60 bg-card/50 overflow-hidden">
      <CardContent className="p-0">
        <div ref={containerRef} className="relative">
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
            <Badge variant="secondary" className="font-mono text-[10px] gap-2">
              <span>Pares matched: {matchedCount}</span>
              {matchedCount > 0 && (
                <>
                  <span className="text-muted-foreground">|</span>
                  <span>Sim. promedio: {(avgSimilarity * 100).toFixed(1)}%</span>
                </>
              )}
            </Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
            <div>
              <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
                Huella Latente
              </p>
              <div className="aspect-square bg-black/95 border border-white/10 rounded-sm flex items-center justify-center overflow-hidden">
                <canvas
                  ref={probeCanvasRef}
                  aria-label={`Huella latente con ${probeMinCount} minucias detectadas`}
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
            <div>
              <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
                Huella Candidata ({props.candidateLabel})
              </p>
              <div className="aspect-square bg-black/95 border border-white/10 rounded-sm flex items-center justify-center overflow-hidden">
                <canvas
                  ref={candidateCanvasRef}
                  aria-label={`Huella candidata ${props.candidateLabel}`}
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          </div>

          <svg
            ref={svgRef}
            aria-hidden="true"
            className="absolute inset-0 w-full h-full pointer-events-none"
          />

          {matchedCount === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground pointer-events-none">
              <Fingerprint className="w-12 h-12 mb-3 opacity-20" />
              <p className="text-sm font-medium">Sin traza de pares</p>
              <p className="text-xs mt-1">
                El candidato no aportó pares coincidentes.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
