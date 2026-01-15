import { CheckCircle, XCircle, Activity, FileText } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import type { FingerprintItem, AppMode } from "@/types/fingerprint";
import { RegistrationForm } from "./RegistrationForm";

interface ResultPanelProps {
  activeMode: AppMode;
  item: FingerprintItem | undefined;
  registration: {
    name: string;
    setName: (val: string) => void;
    id: string;
    setId: (val: string) => void;
  };
}

export function ResultPanel({
  activeMode,
  item,
  registration,
}: ResultPanelProps) {
  const getScoreColor = (score: number) => {
    if (score > 0.8) return "bg-green-500 text-green-500";
    if (score > 0.5) return "bg-yellow-500 text-yellow-500";
    return "bg-red-500 text-red-500";
  };

  return (
    <div className="col-span-3 space-y-4 flex flex-col h-full">
      {activeMode === "register" && (
        <RegistrationForm
          id={registration.id}
          name={registration.name}
          onIdChange={registration.setId}
          onNameChange={registration.setName}
        />
      )}

      {item?.result ? (
        <Card
          className={`flex-1 shadow-md border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden flex flex-col ${
            item.result.matched ? "border-t-4 border-t-green-500" : "border-t-4 border-t-red-500"
          }`}
        >
          <CardHeader className="pb-4 border-b border-border/50 bg-muted/20">
            <CardTitle className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              <FileText className="w-4 h-4" />
              Reporte de Identificación
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6 pt-6 flex-1 overflow-y-auto">
            <div className="flex items-center justify-center py-4">
                {item.result.matched ? (
                    <div className="text-center">
                        <div className="mx-auto w-16 h-16 bg-green-500/10 rounded-full flex items-center justify-center mb-3">
                            <CheckCircle className="w-8 h-8 text-green-500" />
                        </div>
                        <h3 className="text-xl font-bold text-foreground">Identidad Confirmada</h3>
                        <p className="text-sm text-muted-foreground">Coincidencia biométrica exitosa</p>
                    </div>
                ) : (
                    <div className="text-center">
                        <div className="mx-auto w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mb-3">
                            <XCircle className="w-8 h-8 text-red-500" />
                        </div>
                        <h3 className="text-xl font-bold text-foreground">No Identificado</h3>
                        <p className="text-sm text-muted-foreground">No se encontraron coincidencias</p>
                    </div>
                )}
            </div>
            
            {item.result.matched ? (
              <div className="space-y-6">
                {/* Metadatos */}
                <div className="bg-muted/30 p-4 rounded-lg border border-border/50">
                    <div className="grid grid-cols-1 gap-4">
                    <div>
                        <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">
                        Nombre Completo
                        </span>
                        <p className="font-medium text-lg leading-tight mt-1">
                        {item.result.name || "Desconocido"}
                        </p>
                    </div>
                    <div>
                        <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold">
                        Identificador Único (ID)
                        </span>
                        <p className="font-mono text-sm bg-background border border-border p-2 rounded w-full mt-1">
                        {item.result.person_id}
                        </p>
                    </div>
                    </div>
                </div>

                {/* Comparación Visual */}
                {item.referenceImageUrl && (
                  <div className="space-y-3">
                    <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider block">
                      Evidencia Visual
                    </span>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="aspect-square bg-black/5 rounded-lg overflow-hidden border border-border shadow-sm">
                          <img
                            src={
                              item.extractData?.processed_image
                                ? `data:image/png;base64,${item.extractData.processed_image}`
                                : item.preview
                            }
                            className="w-full h-full object-cover"
                            alt="Input"
                          />
                        </div>
                        <p className="text-[10px] text-center font-medium text-muted-foreground uppercase">
                          Huella Entrada
                        </p>
                      </div>
                      <div className="space-y-2">
                        <div className="aspect-square bg-black/5 rounded-lg overflow-hidden border border-border shadow-sm">
                          <img
                            src={item.referenceImageUrl}
                            className="w-full h-full object-cover"
                            alt="Reference"
                          />
                        </div>
                        <p className="text-[10px] text-center font-medium text-muted-foreground uppercase">
                          Registro Base
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Score */}
                <div className="bg-muted/30 p-4 rounded-lg border border-border/50">
                  <div className="flex justify-between items-end mb-2">
                    <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                      Nivel de Confianza
                    </span>
                    <span
                      className={`font-bold ${
                        getScoreColor(item.result.score || 0).split(" ")[1]
                      }`}
                    >
                      {((item.result.score || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-3 w-full bg-background border border-border/50 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-1000 ${
                        getScoreColor(item.result.score || 0).split(" ")[0]
                      }`}
                      style={{
                        width: `${(item.result.score || 0) * 100}%`,
                      }}
                    />
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2 text-right font-mono">
                    Distancia Vectorial (L2): {item.result.distance?.toFixed(4)}
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 bg-muted/30 rounded-lg border border-border/50 text-center">
                    <p className="text-muted-foreground text-sm">
                    El sistema no encontró registros que coincidan con los parámetros biométricos proporcionados.
                    </p>
                </div>
                {item.result.score && item.result.score > 0 && (
                  <div className="p-4 bg-red-500/5 border border-red-500/20 rounded-lg">
                    <p className="text-xs font-semibold text-red-500 uppercase tracking-wider mb-2">Mejor coincidencia rechazada</p>
                    <div className="flex justify-between items-center bg-background/50 p-2 rounded">
                      <span className="font-mono text-xs text-foreground">
                        {item.result.person_id || "?"}
                      </span>
                      <span className="text-xs font-bold text-red-500">
                        {(item.result.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-2">
                        El puntaje no superó el umbral de seguridad establecido.
                    </p>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      ) : (
        <Card className="flex-1 flex items-center justify-center text-muted-foreground border-dashed bg-muted/5">
          <div className="text-center opacity-50">
            <Activity className="w-12 h-12 mx-auto mb-3" />
            <p className="text-sm font-medium">Esperando resultados</p>
            <p className="text-xs mt-1">Procesa una huella para ver el reporte</p>
          </div>
        </Card>
      )}
    </div>
  );
}
