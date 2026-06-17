import { useEffect } from "react";
import { Pencil, Plus, Trash2, Move, X, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { useCanvasDrawer, type EditingMode } from "@/hooks/useCanvasDrawer";
import type { MinutiaPoint } from "@/lib/api";

interface MinutiaeEditorProps {
  imageUrl: string;
  initialMinutiae: MinutiaPoint[];
  onSave: (minutiae: MinutiaPoint[]) => void;
  onCancel: () => void;
}

interface ModeButtonConfig {
  mode: EditingMode;
  label: string;
  icon: typeof Plus;
  description: string;
}

const MODE_BUTTONS: ModeButtonConfig[] = [
  {
    mode: "view",
    label: "Ver",
    icon: Pencil,
    description: "Solo visualización",
  },
  {
    mode: "add",
    label: "Añadir",
    icon: Plus,
    description: "Click para añadir. Mayús+click = Bifurcación",
  },
  {
    mode: "delete",
    label: "Eliminar",
    icon: Trash2,
    description: "Click en una minucia para eliminarla",
  },
  {
    mode: "move",
    label: "Mover",
    icon: Move,
    description: "Arrastra una minucia para moverla",
  },
];

export function MinutiaeEditor({
  imageUrl,
  initialMinutiae,
  onSave,
  onCancel,
}: MinutiaeEditorProps) {
  const {
    canvasRef,
    editingState,
    setMode,
    setMinutiae,
    handleCanvasClick,
    handleMouseMove,
    handleSave,
  } = useCanvasDrawer(null, imageUrl, { minutiae: initialMinutiae });

  // Sync initial minutiae into the hook when they change
  useEffect(() => {
    setMinutiae(initialMinutiae);
  }, [initialMinutiae, setMinutiae]);

  const handleSaveClick = () => {
    const edited = handleSave();
    onSave(edited);
  };

  const currentMode = editingState.mode;
  const modeDescription =
    MODE_BUTTONS.find((b) => b.mode === currentMode)?.description ?? "";

  const terminations = editingState.minutiae.filter(
    (m) => m.type === 0
  ).length;
  const bifurcations = editingState.minutiae.filter(
    (m) => m.type === 1
  ).length;

  return (
    <Card className="border-border/50 bg-black/95 shadow-2xl overflow-hidden">
      <CardHeader className="border-b border-border/50 bg-muted/20 pb-3">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          <Pencil className="w-4 h-4" />
          Editor de Minucias
        </CardTitle>
      </CardHeader>

      <CardContent className="p-0 relative">
        {/* Toolbar */}
        <div className="absolute top-3 left-3 right-3 z-10 flex items-center gap-2">
          {/* Mode buttons */}
          <div className="flex gap-1 bg-black/80 backdrop-blur-sm rounded-lg border border-white/10 p-1">
            {MODE_BUTTONS.map((btn) => (
              <Button
                key={btn.mode}
                size="sm"
                variant={
                  currentMode === btn.mode ? "default" : "ghost"
                }
                className={`h-8 px-2 gap-1.5 text-xs ${
                  currentMode === btn.mode
                    ? ""
                    : "text-white/60 hover:text-white hover:bg-white/10"
                }`}
                onClick={() => setMode(btn.mode)}
                title={btn.description}
              >
                <btn.icon className="w-3.5 h-3.5" />
                {btn.label}
              </Button>
            ))}
          </div>

          <div className="flex-1" />

          {/* Minutiae stats */}
          <div className="flex items-center gap-3 bg-black/80 backdrop-blur-sm rounded-lg border border-white/10 px-3 py-1.5 text-xs">
            <span className="text-white/60">Total:</span>
            <span className="text-white font-mono font-bold">
              {editingState.minutiae.length}
            </span>
            <span className="w-px h-4 bg-white/10" />
            <span className="text-red-400 font-mono">{terminations}</span>
            <span className="text-white/40">/</span>
            <span className="text-green-400 font-mono">{bifurcations}</span>
          </div>
        </div>

        {/* Mode description tooltip */}
        <div className="absolute top-14 left-3 z-10">
          <div className="bg-black/70 backdrop-blur-sm rounded-md border border-white/10 px-2.5 py-1 text-[10px] text-white/50">
            {modeDescription}
          </div>
        </div>

        {/* Canvas area */}
        <div className="w-full bg-black/90 flex items-center justify-center min-h-[400px]">
          {/* Grid background effect */}
          <div
            className="absolute inset-0 opacity-20 pointer-events-none"
            style={{
              backgroundImage:
                "linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)",
              backgroundSize: "20px 20px",
            }}
          />

          <div className="relative p-8">
            <canvas
              ref={canvasRef}
              className="max-w-full max-h-[500px] object-contain rounded-sm border border-white/10 cursor-crosshair"
              onClick={handleCanvasClick}
              onMouseMove={handleMouseMove}
            />
          </div>
        </div>

        {/* Legend */}
        <div className="absolute bottom-20 left-3 z-10">
          <div className="bg-black/80 backdrop-blur-sm rounded-lg border border-white/10 px-3 py-2 text-[11px] space-y-1.5">
            <div className="flex items-center gap-2 text-white/50 uppercase tracking-wider text-[10px] font-semibold">
              <span>Leyenda</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-red-500 border border-white/30 inline-block" />
              <span className="text-white/70">Terminación</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-green-500 border border-white/30 inline-block" />
              <span className="text-white/70">Bifurcación</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full border-2 border-yellow-400 inline-block" />
              <span className="text-white/70">Seleccionada</span>
            </div>
          </div>
        </div>
      </CardContent>

      <CardFooter className="border-t border-border/50 bg-muted/20 p-3 flex justify-between">
        <Button
          variant="ghost"
          onClick={onCancel}
          className="text-white/60 hover:text-white gap-2"
        >
          <X className="w-4 h-4" />
          Cancelar
        </Button>

        <Button
          variant="default"
          onClick={handleSaveClick}
          className="gap-2"
        >
          <Save className="w-4 h-4" />
          Guardar &amp; Re-buscar
        </Button>
      </CardFooter>
    </Card>
  );
}
