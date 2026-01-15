import { useRef } from "react";
import { Plus, Loader2, CheckCircle, XCircle, FileImage } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { FingerprintItem, AppMode } from "@/types/fingerprint";

interface FingerprintListProps {
  fingerprints: FingerprintItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAddFiles: (files: FileList | null) => void;
  onProcessAll: () => void;
  loading: boolean;
  activeMode: AppMode;
  registrationValid: boolean;
}

export function FingerprintList({
  fingerprints,
  selectedId,
  onSelect,
  onAddFiles,
  onProcessAll,
  loading,
  activeMode,
  registrationValid,
}: FingerprintListProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onAddFiles(e.target.files);
  };

  return (
    <div className="col-span-3 flex flex-col h-full">
      <Card className="flex flex-col h-full shadow-sm border-border/50 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-3 border-b border-border/50 bg-muted/20">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
              <FileImage className="w-4 h-4" />
              Cola de Procesamiento
            </CardTitle>
            <Badge variant="outline" className="font-mono text-xs">
              {fingerprints.length}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden flex flex-col p-0">
          {fingerprints.length === 0 ? (
            <div
              className="flex-1 flex flex-col items-center justify-center p-6 text-center border-2 border-dashed border-muted m-4 rounded-xl cursor-pointer hover:border-primary/50 hover:bg-muted/50 transition-all"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="p-4 rounded-full bg-primary/10 mb-3 text-primary">
                <Plus className="w-6 h-6" />
              </div>
              <p className="text-sm font-medium">Arrastra huellas aquí</p>
              <p className="text-xs text-muted-foreground mt-1">
                o haz clic para explorar
              </p>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {fingerprints.map((fp) => (
                <div
                  key={fp.id}
                  onClick={() => onSelect(fp.id)}
                  className={`group p-3 rounded-lg border transition-all flex items-center gap-3 cursor-pointer relative overflow-hidden ${
                    selectedId === fp.id
                      ? "border-primary bg-primary/5 shadow-sm"
                      : "border-transparent hover:bg-muted/50 hover:border-border"
                  }`}
                >
                  {selectedId === fp.id && (
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary" />
                  )}

                  <div className="w-10 h-10 rounded bg-muted flex-shrink-0 overflow-hidden border border-border">
                    <img
                      src={fp.preview}
                      className="w-full h-full object-cover opacity-90"
                      alt="Preview"
                    />
                  </div>

                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-medium truncate ${
                        selectedId === fp.id
                          ? "text-primary"
                          : "text-foreground"
                      }`}
                    >
                      {fp.file.name}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      {fp.status === "pending" && (
                        <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                          <div className="w-1.5 h-1.5 rounded-full bg-yellow-500" />{" "}
                          Pendiente
                        </span>
                      )}
                      {fp.status === "processing" && (
                        <span className="text-[10px] text-blue-500 flex items-center gap-1">
                          <Loader2 className="w-3 h-3 animate-spin" />{" "}
                          Procesando...
                        </span>
                      )}
                      {fp.status === "completed" && (
                        <span
                          className={`text-[10px] flex items-center gap-1 ${
                            fp.result?.matched
                              ? "text-green-600"
                              : "text-red-500"
                          }`}
                        >
                          {fp.result?.matched ? (
                            <>
                              <CheckCircle className="w-3 h-3" /> Identificado
                            </>
                          ) : (
                            <>
                              <XCircle className="w-3 h-3" /> No match
                            </>
                          )}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            multiple
            onChange={handleFileChange}
          />
        </CardContent>

        <CardFooter className="p-3 border-t border-border/50 bg-muted/20">
          {fingerprints.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              className="w-full mb-2 mr-2"
              onClick={() => fileInputRef.current?.click()}
            >
              <Plus className="w-4 h-4 mr-2" /> Agregar más
            </Button>
          )}
          <Button
            className="w-full shadow-lg shadow-primary/20"
            onClick={onProcessAll}
            disabled={
              loading ||
              fingerprints.length === 0 ||
              (activeMode === "register" && !registrationValid)
            }
          >
            {loading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              "Iniciar Procesamiento"
            )}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
