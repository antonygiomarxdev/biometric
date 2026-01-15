import { Fingerprint, Scan, Maximize2, Loader2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { FingerprintItem } from "@/types/fingerprint";
import { useCanvasDrawer } from "@/hooks/useCanvasDrawer";

interface FingerprintViewerProps {
  item: FingerprintItem | undefined;
}

export function FingerprintViewer({ item }: FingerprintViewerProps) {
  const canvasRef = useCanvasDrawer(
    item?.id || null,
    item?.preview,
    item?.extractData
      ? {
          minutiae: item.extractData.minutiae,
          processed_image: item.extractData.processed_image,
        }
      : undefined
  );

  return (
    <div className="col-span-6 flex flex-col h-full">
      <Card className="flex flex-col h-full shadow-md border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
        <CardHeader className="border-b border-border/50 bg-muted/20 pb-3">
          <div className="flex justify-between items-center">
            <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              <Scan className="w-4 h-4" />
              Visualizador Biométrico
            </CardTitle>
            {item?.extractData && (
              <div className="flex gap-2">
                <Badge
                  variant="secondary"
                  className="font-mono text-[10px] tracking-tight"
                >
                  {item.extractData.image_shape
                    ? `${item.extractData.image_shape[0]}x${item.extractData.image_shape[1]}px`
                    : "N/A"}
                </Badge>
                <Badge
                  variant="outline"
                  className="font-mono text-[10px] tracking-tight"
                >
                  {item.extractData.image_dtype || "N/A"}
                </Badge>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="flex-1 p-0 relative bg-black/90 flex items-center justify-center overflow-hidden">
          {/* Grid Background Effect */}
          <div
            className="absolute inset-0 opacity-20 pointer-events-none"
            style={{
              backgroundImage:
                "linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)",
              backgroundSize: "20px 20px",
            }}
          />

          {item ? (
            <div className="relative w-full h-full flex items-center justify-center p-8">
              <div className="relative shadow-2xl shadow-primary/20 transition-all duration-500 ease-out">
                {/* Scanner Line Effect */}
                {item.status === "processing" && (
                  <>
                    <div className="absolute inset-0 z-10 pointer-events-none overflow-hidden rounded-lg bg-primary/5">
                      <div className="w-full h-1 bg-primary/50 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-[scan_2s_ease-in-out_infinite]" />
                    </div>
                    <div className="absolute inset-0 z-20 flex items-center justify-center">
                      <div className="bg-black/70 backdrop-blur-sm px-4 py-2 rounded-full border border-primary/30 flex items-center gap-2">
                        <Loader2 className="w-4 h-4 text-primary animate-spin" />
                        <span className="text-xs font-medium text-white tracking-wide">
                          ANALIZANDO...
                        </span>
                      </div>
                    </div>
                  </>
                )}

                <canvas
                  ref={canvasRef}
                  className="max-w-full max-h-[500px] object-contain rounded-sm border border-white/10"
                />
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground z-10">
              <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-white/5 flex items-center justify-center border border-white/10 animate-pulse">
                <Fingerprint className="w-12 h-12 opacity-20" />
              </div>
              <p className="font-medium">Esperando entrada biométrica</p>
              <p className="text-sm opacity-50 mt-1">
                Selecciona un archivo de la cola
              </p>
            </div>
          )}

          {/* Overlay de estadísticas */}
          {item?.extractData && (
            <div className="absolute bottom-6 left-6 right-6 animate-in slide-in-from-bottom-4 duration-500">
              <div className="bg-black/60 backdrop-blur-md border border-white/10 rounded-xl p-4 grid grid-cols-3 gap-6 text-xs text-white shadow-xl">
                <div>
                  <span className="text-white/50 block mb-1 uppercase tracking-wider text-[10px]">
                    Minucias
                  </span>
                  <div className="flex items-baseline gap-1">
                    <span className="text-2xl font-bold tracking-tight">
                      {item.extractData.minutiae_count}
                    </span>
                    <span className="text-white/40">
                      / {item.extractData.minutiae_initial_count || "?"}
                    </span>
                  </div>
                </div>
                <div className="border-l border-white/10 pl-6">
                  <span className="text-white/50 block mb-1 uppercase tracking-wider text-[10px]">
                    Terminaciones
                  </span>
                  <span className="text-2xl font-bold text-red-400 tracking-tight">
                    {item.extractData.terminations}
                  </span>
                </div>
                <div className="border-l border-white/10 pl-6">
                  <span className="text-white/50 block mb-1 uppercase tracking-wider text-[10px]">
                    Bifurcaciones
                  </span>
                  <span className="text-2xl font-bold text-green-400 tracking-tight">
                    {item.extractData.bifurcations}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Zoom/Controls placeholder */}
          {item && (
            <div className="absolute top-4 right-4 flex flex-col gap-2">
              <Button
                size="icon"
                variant="secondary"
                className="h-8 w-8 bg-black/50 hover:bg-black/70 border border-white/10 text-white transition-colors"
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
