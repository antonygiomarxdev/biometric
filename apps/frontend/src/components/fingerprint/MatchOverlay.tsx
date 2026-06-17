import { useRef } from "react";
import { Fingerprint } from "lucide-react";
import { useMatchCanvas, type UseMatchCanvasArgs } from "@/hooks/useMatchCanvas";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface MatchOverlayProps extends UseMatchCanvasArgs {
  /** Display label for the candidate canvas caption (e.g. external_id or full_name). */
  candidateLabel: string;
}

export function MatchOverlay(props: MatchOverlayProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const { probeCanvasRef, candidateCanvasRef, svgRef } = useMatchCanvas({
    probeImageUrl: props.probeImageUrl,
    probeMinutiae: props.probeMinutiae,
    candidateImageUrl: props.candidateImageUrl,
    candidateMinutiae: props.candidateMinutiae,
    matchTrace: props.matchTrace,
    containerRef,
  });

  const matchedCount = props.matchTrace.length;
  const avgSimilarity =
    matchedCount > 0
      ? props.matchTrace.reduce((s, e) => s + e.similarity, 0) / matchedCount
      : 0;
  const probeMinCount = props.probeMinutiae.length;

  return (
    <Card className="border-border/60 bg-card/50 overflow-hidden">
      <CardContent className="p-0">
        <div
          ref={containerRef}
          className="relative"
        >
          {/* Stats badge — top center, between the two canvas captions */}
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
            <Badge
              variant="secondary"
              className="font-mono text-[10px] gap-2"
            >
              <span>Pares matched: {matchedCount}</span>
              {matchedCount > 0 && (
                <>
                  <span className="text-muted-foreground">|</span>
                  <span>Sim. promedio: {(avgSimilarity * 100).toFixed(1)}%</span>
                </>
              )}
            </Badge>
          </div>

          {/* Two canvases side by side */}
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

          {/* SVG line overlay layer — pointer-events-none so it doesn't block future click handlers */}
          <svg
            ref={svgRef}
            aria-hidden="true"
            className="absolute inset-0 w-full h-full pointer-events-none"
          />

          {/* Empty state when no trace */}
          {matchedCount === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground pointer-events-none">
              <Fingerprint className="w-12 h-12 mb-3 opacity-20" />
              <p className="text-sm font-medium">Sin traza de cilindros</p>
              <p className="text-xs mt-1">
                El candidato no aportó cilindros coincidentes.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
