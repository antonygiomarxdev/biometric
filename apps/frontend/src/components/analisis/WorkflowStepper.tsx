import React from 'react';
import { cn } from "@/lib/utils";
import { Upload, Fingerprint, Search, CheckCircle2, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";

interface WorkflowStepperProps {
  current: 0 | 1 | 2;
  searchRunning: boolean;
  minutiae: number;
  candidateCount: number;
  queryTimeMs: number | undefined;
  children: React.ReactNode;
}

export function WorkflowStepper({
  current,
  searchRunning,
  minutiae,
  candidateCount,
  queryTimeMs,
  children,
}: WorkflowStepperProps): React.JSX.Element {
  const steps: Array<{
    n: 1 | 2 | 3 | 4;
    label: string;
    icon: React.ReactNode;
    status: string;
    running: boolean;
  }> = [
    {
      n: 1,
      label: "Subir",
      icon: <Upload className="w-4 h-4" />,
      status: current >= 1 ? "Imagen cargada" : "Esperando archivo",
      running: false,
    },
    {
      n: 2,
      label: "Extraer",
      icon: <Fingerprint className="w-4 h-4" />,
      status: minutiae > 0
        ? `${minutiae} minucias detectadas`
        : current < 1
        ? "Esperando imagen"
        : "Listo para procesar",
      running: false,
    },
    {
      n: 3,
      label: "Buscar",
      icon: <Search className="w-4 h-4" />,
      status: searchRunning
        ? "KNN sobre Qdrant…"
        : queryTimeMs !== undefined
        ? `${candidateCount} candidato${candidateCount !== 1 ? "s" : ""} · ${queryTimeMs}ms`
        : current < 2
        ? "Esperando extracción"
        : "Listo para buscar",
      running: searchRunning,
    },
    {
      n: 4,
      label: "Resultado",
      icon: <CheckCircle2 className="w-4 h-4" />,
      status:
        current >= 2
          ? "Comparación activa"
          : "Pendiente",
      running: false,
    },
  ];

  const progressPct = (current / 3) * 100;

  return (
    <Card className="overflow-hidden border-border/60">
      <div className="h-1.5 bg-muted overflow-hidden">
        <div
          className={cn(
            "h-full transition-all duration-700 ease-out",
            searchRunning
              ? "bg-primary/60 animate-pulse"
              : "bg-primary",
          )}
          style={{ width: `${progressPct}%` }}
        />
      </div>
      <div className="flex items-stretch p-2 bg-card">
        {steps.map((s, i) => {
          const done = current >= s.n;
          const active = current === s.n - 1;
          const isLast = i === steps.length - 1;
          return (
            <div key={s.n} className="flex items-stretch flex-1 min-w-0">
              <div
                className={cn(
                  "flex items-center gap-3 flex-1 min-w-0 px-3 py-2 rounded transition-all",
                  done && "bg-primary/10",
                  active && !done && "bg-primary/5 ring-1 ring-primary/40",
                )}
              >
                <div
                  className={cn(
                    "flex items-center justify-center w-9 h-9 rounded-full border-2 transition-all flex-shrink-0",
                    done && "bg-primary border-primary text-primary-foreground shadow-sm",
                    active &&
                      !done &&
                      "border-primary text-primary shadow-md shadow-primary/30",
                    !active && !done && "border-border text-muted-foreground",
                    active && !done && s.running && "animate-pulse",
                  )}
                >
                  {done ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : s.running ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    s.icon
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className="text-[10px] font-mono text-muted-foreground">
                      {String(s.n).padStart(2, "0")}
                    </span>
                    <span
                      className={cn(
                        "text-sm font-semibold truncate",
                        done && "text-primary",
                        active && !done && "text-foreground",
                        !active && !done && "text-muted-foreground",
                      )}
                    >
                      {s.label}
                    </span>
                  </div>
                  <div className="text-[11px] text-muted-foreground truncate">
                    {s.status}
                  </div>
                </div>
              </div>
              {!isLast && (
                <div
                  className={cn(
                    "w-1 self-stretch mx-0.5 transition-colors rounded",
                    current > s.n ? "bg-primary" : "bg-border",
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
      <div className="p-4 bg-background/30">{children}</div>
    </Card>
  );
}

interface UploadDropzoneProps {
  onFile: (file: File) => void;
  running?: boolean;
}

export function UploadDropzone({
  onFile,
  running = false,
}: UploadDropzoneProps): React.JSX.Element {
  return (
    <label
      className={cn(
        "block cursor-pointer transition-all",
        running && "opacity-50 pointer-events-none",
      )}
    >
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-8 transition-all",
          "border-border hover:border-primary hover:bg-primary/5",
        )}
      >
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
            <Upload className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-lg font-semibold mb-1">
            Subí la huella latente
          </h2>
          <p className="text-sm text-muted-foreground">
            Arrastrá una imagen o hacé click para seleccionar.
            <br />
            BMP, PNG o JPEG. Máx 10MB.
          </p>
        </div>
      </div>
      <input
        type="file"
        className="hidden"
        accept="image/*"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
    </label>
  );
}
