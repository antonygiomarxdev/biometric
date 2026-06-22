import { useRef, useState, useCallback } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Fingerprint,
  Upload,
  CheckCircle,
  ArrowLeft,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import {
  listPersons,
  createFingerprintSlot,
  enrollFingerprint,
} from "@/lib/api";
import type { PersonResponse, CaptureResponse } from "@/lib/api";

type EnrollStep = "select-person" | "upload-image" | "submitting" | "done";

const VALID_TYPES = ["image/bmp", "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/tif", "image/x-tiff"] as const;
const MAX_BYTES = 10 * 1024 * 1024;
const STEPS: { id: EnrollStep; label: string }[] = [
  { id: "select-person", label: "1. Seleccionar persona" },
  { id: "upload-image", label: "2. Subir imagen de huella" },
  { id: "submitting", label: "3. Indexando embedding" },
  { id: "done", label: "4. Huella enrolada" },
];

function StepIndicator({ current }: { current: EnrollStep }): React.JSX.Element {
  const currentIdx = STEPS.findIndex((s) => s.id === current);
  return (
    <div className="flex items-center gap-1" aria-label="Indicador de pasos">
      {STEPS.slice(0, 3).map((s, i) => {
        const isActive = i === currentIdx;
        const isPast = i < currentIdx;
        return (
          <div
            key={s.id}
            className={`h-2 w-2 rounded-full ${
              isActive || isPast ? "bg-primary" : "bg-muted"
            }`}
            aria-current={isActive ? "step" : undefined}
            aria-label={`Paso ${i + 1} ${i + 1 === currentIdx + 1 ? "(actual)" : isPast ? "(completado)" : "(pendiente)"}`}
          />
        );
      })}
    </div>
  );
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (ev) => resolve(ev.target?.result as string);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

export default function EnrollPage(): React.JSX.Element {
  const { caseId } = useParams<{ caseId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { addToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [step, setStep] = useState<EnrollStep>("select-person");
  const [selectedPersonId, setSelectedPersonId] = useState<string | null>(null);
  const [fingerprintSlotId, setFingerprintSlotId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [enrolledCapture, setEnrolledCapture] = useState<CaptureResponse | null>(null);

  const {
    data: persons,
    isLoading: personsLoading,
    isError: personsError,
  } = useQuery({
    queryKey: ["persons"],
    queryFn: () => listPersons(0, 100),
  });

  const slotMutation = useMutation({
    mutationFn: (personId: string) => createFingerprintSlot(personId, 0, "rolled"),
    onSuccess: (slot) => {
      setFingerprintSlotId(slot.id);
      setStep("upload-image");
    },
    onError: (err: Error) => {
      addToast({
        type: "error",
        title: "No se pudo crear el slot de huella",
        description: err.message,
      });
    },
  });

  const handlePersonChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const pid = e.target.value;
    setSelectedPersonId(pid);
    if (pid) {
      slotMutation.mutate(pid);
    }
  };

  const enrollMutation = useMutation({
    mutationFn: async () => {
      if (!fingerprintSlotId || !file) {
        throw new Error("Falta slot o archivo");
      }
      return enrollFingerprint(fingerprintSlotId, file);
    },
    onSuccess: (capture) => {
      setEnrolledCapture(capture);
      setStep("done");
      addToast({
        type: "success",
        title: "Huella enrolada",
        description: "Captura registrada con embedding AFR-Net",
      });
      queryClient.invalidateQueries({ queryKey: ["persons"] });
      if (caseId) {
        queryClient.invalidateQueries({ queryKey: ["evidencias", caseId] });
      }
    },
    onError: (err: Error) => {
      addToast({
        type: "error",
        title: "No se pudo procesar la imagen",
        description: err.message,
      });
      setStep("upload-image");
    },
  });

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (!f) return;
      if (!VALID_TYPES.includes(f.type as (typeof VALID_TYPES)[number])) {
        addToast({
          type: "error",
          title: "Tipo de archivo inválido",
          description: "Selecciona una imagen BMP, PNG, JPEG o TIFF",
        });
        return;
      }
      if (f.size > MAX_BYTES) {
        addToast({
          type: "error",
          title: "Archivo demasiado grande",
          description: "El archivo no debe exceder 10MB",
        });
        return;
      }
      setFile(f);
      try {
        const dataUrl = await readFileAsDataUrl(f);
        setPreviewUrl(dataUrl);
        setStep("submitting");
        await enrollMutation.mutateAsync();
      } catch (err) {
        addToast({
          type: "error",
          title: "No se pudo procesar la imagen",
          description: err instanceof Error ? err.message : "Error desconocido",
        });
      }
    },
    [addToast, enrollMutation],
  );

  const handleEnrollAnother = () => {
    setSelectedPersonId(null);
    setFingerprintSlotId(null);
    setFile(null);
    setPreviewUrl(null);
    setEnrolledCapture(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setStep("select-person");
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-8 font-sans dark">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex items-center justify-between border-b border-border pb-6">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate(caseId ? `/cases/${caseId}/compare` : "/")}
            >
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Enrolar Huella</h1>
              <p className="text-sm text-muted-foreground">
                Selecciona una persona y sube una imagen. El sistema genera automáticamente
                el embedding AFR-Net y lo indexa en la base vectorial.
              </p>
            </div>
          </div>
          <StepIndicator current={step} />
        </header>

        <main>
          {step === "select-person" && (
            <Card className="border-border/60">
              <CardHeader>
                <CardTitle className="text-base">1. Seleccionar persona</CardTitle>
                <CardDescription>
                  Elige la persona a la que se asociará esta huella.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {personsLoading && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Cargando personas...</span>
                  </div>
                )}
                {personsError && (
                  <div className="flex items-center gap-2 text-destructive">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm">No se pudo cargar la lista de personas.</span>
                  </div>
                )}
                {persons && persons.length === 0 && (
                  <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                    <Fingerprint className="w-12 h-12 mb-3 opacity-20" />
                    <p className="text-sm font-medium">No hay personas registradas</p>
                    <p className="text-xs mt-1 text-center max-w-md">
                      Ejecute <code className="font-mono">python scripts/seed_socofing.py</code>{" "}
                      para cargar el dataset SOCOFing.
                    </p>
                  </div>
                )}
                {persons && persons.length > 0 && (
                  <div>
                    <label className="text-sm font-medium" htmlFor="person-select">
                      Persona:
                    </label>
                    <select
                      id="person-select"
                      className="mt-1 w-full bg-muted/30 border border-border rounded-md px-3 py-2 text-sm"
                      value={selectedPersonId ?? ""}
                      onChange={handlePersonChange}
                      disabled={slotMutation.isPending}
                    >
                      <option value="">— Selecciona una persona —</option>
                      {persons.map((p: PersonResponse) => (
                        <option key={p.id} value={p.id}>
                          {p.full_name ?? p.external_id ?? p.id}
                          {p.external_id ? ` (${p.external_id})` : ""}
                        </option>
                      ))}
                    </select>
                    {slotMutation.isPending && (
                      <div className="flex items-center gap-2 text-muted-foreground mt-2">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span className="text-xs">Creando slot de huella...</span>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {step === "upload-image" && (
            <Card className="border-border/60">
              <CardHeader>
                <CardTitle className="text-base">2. Subir imagen de huella</CardTitle>
                <CardDescription>
                  Selecciona una imagen (BMP, PNG, JPEG — máx 10MB). El modelo
                  AFR-Net procesará la imagen y generará el embedding automáticamente.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary/50 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-12 h-12 mb-3 opacity-40" />
                  <p className="text-sm font-medium">Subir huella</p>
                  <p className="text-xs mt-1 text-muted-foreground">
                    BMP, PNG, JPEG — máx 10MB
                  </p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png, image/jpeg, image/bmp, image/tiff"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </CardContent>
            </Card>
          )}

          {step === "submitting" && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-primary mb-3" />
                <p className="text-sm font-medium">Generando embedding AFR-Net...</p>
                <p className="text-xs mt-1 text-muted-foreground">
                  La huella se está indexando en la base de datos vectorial.
                </p>
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Vista previa de la huella"
                    className="mt-4 max-h-48 rounded border border-border"
                  />
                )}
              </CardContent>
            </Card>
          )}

          {step === "done" && enrolledCapture && (
            <Card className="border-border/60">
              <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                <CheckCircle className="w-12 h-12 text-green-500 mb-4" />
                <h2 className="text-xl font-semibold tracking-tight">
                  Huella enrolada exitosamente
                </h2>
                <p className="text-sm text-muted-foreground mt-2 max-w-md">
                  Captura registrada (algoritmo {enrolledCapture.algorithm_version})
                  y embedding indexado en la base de datos vectorial.
                </p>
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Huella enrolada"
                    className="mt-4 max-h-48 rounded border border-border"
                  />
                )}
                <div className="flex items-center gap-2 mt-6">
                  <Button variant="outline" onClick={handleEnrollAnother}>
                    <Fingerprint className="w-4 h-4 mr-2" />
                    Enrolar otra huella
                  </Button>
                  <Button onClick={() => navigate(caseId ? `/cases/${caseId}/compare` : "/")}>
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Volver al panel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </div>
  );
}
