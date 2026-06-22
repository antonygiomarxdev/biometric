import { useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { UploadDropzone } from "@/components/analisis/WorkflowStepper";
import { useCanvas } from "@/hooks/useCanvas";
import type { MatchSearchResponse } from "@/lib/api";

interface ProbePanelProps {
  probeDataUrl: string | null;
  searchResult: MatchSearchResponse | null;
  onFile: (file: File) => void;
  isLoading: boolean;
}

/**
 * Probe preview: the original uploaded image, full-width.
 *
 * GradCAM is computed by the backend on every search and returned in
 * the response (for debugging and for the model explainability story),
 * but the perito does not see it in the main workflow — they said
 * it's not useful, and it tends to mislead when the preprocessor
 * has issues (the heatmap activates on the empty border, not on the
 * fingerprint).  The GradCAM is still available at
 *   GET /api/v1/admin/searches/{id}/gradcam
 * for engineers debugging the model.  See:
 *   docs/LESSONS_LEARNED.md §"Phase 29: AFIS Quality Pre-Requisites"
 *   .planning/STATE.md §"Phase 29 — Deep Embedding"
 */
export function ProbePanel({
  probeDataUrl,
  searchResult: _searchResult,
  onFile,
  isLoading,
}: ProbePanelProps) {
  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useCanvas(originalCanvasRef, probeDataUrl);

  return (
    <Card className="border-border/60">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
          Original
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center max-w-[480px] mx-auto">
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
  );
}
