import { useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { UploadDropzone } from "@/components/analisis/WorkflowStepper";
import { drawMinutiaMarker } from "@/hooks/useMatchCanvas";
import { useCanvas } from "@/hooks/useCanvas";

const PALETTE_HIT = "#ffffff";

interface ProbePanelProps {
  probeDataUrl: string | null;
  probePreviewUrl: string | null;
  probeMinutiae: any[];
  searchResult: any;
  selectedCandidate: any;
  onFile: (file: File) => void;
  isLoading: boolean;
}

export function ProbePanel({
  probeDataUrl,
  probePreviewUrl,
  probeMinutiae,
  searchResult,
  selectedCandidate,
  onFile,
  isLoading,
}: ProbePanelProps) {
  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const processedCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useCanvas(originalCanvasRef, probeDataUrl);

  useCanvas(processedCanvasRef, probePreviewUrl, (ctx, img) => {
    const matchedIndices = new Set<number>();
    if (selectedCandidate) {
      for (const e of selectedCandidate.supporting_pairs) {
        matchedIndices.add(e.probe_mi_idx);
      }
    }

    const minutiaeToDraw = searchResult?.probe_minutiae ?? probeMinutiae;
    if (minutiaeToDraw) {
      for (let i = 0; i < minutiaeToDraw.length; i++) {
        const m = minutiaeToDraw[i];
        if (!m) continue;
        const color = matchedIndices.has(i)
          ? PALETTE_HIT
          : "rgba(255,255,255,0.7)";
        drawMinutiaMarker(ctx, m, color);
      }
    }
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card className="border-border/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
            Original
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3">
          <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center">
            {probeDataUrl ? (
              <canvas
                ref={originalCanvasRef}
                className="w-full h-full object-contain"
              />
            ) : (
              <UploadDropzone onFile={onFile} running={isLoading} />
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="border-border/60">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
            Procesada · Minucias Detectadas
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3">
          <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center">
            {probePreviewUrl ? (
              <canvas
                ref={processedCanvasRef}
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-center text-muted-foreground text-sm px-4">
                Subí una huella para empezar
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
