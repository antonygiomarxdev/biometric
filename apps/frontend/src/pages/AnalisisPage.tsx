import { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Search,
  Upload,
  Fingerprint,
  Loader2,
  XCircle,
  CheckCircle2,
  ArrowLeft,
  FilePlus,
  UserPlus,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { CandidateCard } from "@/components/fingerprint/CandidateCard";
import { CandidateDetailPanel } from "@/components/fingerprint/CandidateDetailPanel";
import {
  searchMatching,
  getMinutiaeForImage,
  listPersons,
  createFingerprintSlot,
  enrollFingerprint,
  createCase,
  type MatchCandidate,
  type MatchSearchResponse,
  type MinutiaPoint,
  type FingerprintPreviewResponse,
  type CaseCreateInput,
} from "@/lib/api";

const VALID_TYPES = ["image/bmp", "image/png", "image/jpeg", "image/jpg"];
const MAX_BYTES = 10 * 1024 * 1024;

type Step = "upload" | "analyze" | "result" | "decide";

export default function AnalisisPage() {
  const navigate = useNavigate();
  const { addToast } = useToast();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const [latentFile, setLatentFile] = useState<File | null>(null);
  const [latentPreview, setLatentPreview] = useState<string | null>(null);
  const [step, setStep] = useState<Step>("upload");
  const [preview, setPreview] = useState<FingerprintPreviewResponse | null>(null);
  const [searchResult, setSearchResult] = useState<MatchSearchResponse | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<MatchCandidate | null>(null);

  // Enrollment mode: pick a person + their slot
  const [enrollPersonId, setEnrollPersonId] = useState<string>("");
  const [showEnrollPicker, setShowEnrollPicker] = useState(false);
  const [enrolling, setEnrolling] = useState(false);

  // Create-case mode
  const [showCreateCase, setShowCreateCase] = useState(false);
  const [caseNumber, setCaseNumber] = useState("");
  const [caseTitle, setCaseTitle] = useState("");

  const previewMutation = useMutation({
    mutationFn: (file: File) => getMinutiaeForImage(file),
    onSuccess: (result) => {
      setPreview(result);
      setStep("analyze");
    },
    onError: (err: Error) => {
      addToast({ type: "error", title: "Error al procesar", description: err.message });
    },
  });

  const searchMutation = useMutation({
    mutationFn: (file: File) => searchMatching(file, 10),
    onSuccess: (result) => {
      setSearchResult(result);
      setSelectedCandidate(result.candidates[0] ?? null);
      setStep("result");
      addToast({
        type: result.candidates.length > 0 ? "success" : "info",
        title: result.candidates.length > 0
          ? `${result.candidates.length} candidato${result.candidates.length !== 1 ? "s" : ""}`
          : "Sin coincidencias",
        description: result.candidates.length > 0
          ? "Revisa los candidatos abajo"
          : "Puedes enrolar esta huella si pertenece a alguien nuevo",
        duration: 4000,
      });
    },
    onError: (err: Error) => {
      addToast({ type: "error", title: "Error en búsqueda", description: err.message });
    },
  });

  const enrollMutation = useMutation({
    mutationFn: async ({ personId, file }: { personId: string; file: File }) => {
      const slot = await createFingerprintSlot(personId, 0, "rolled");
      return enrollFingerprint(slot.id, file);
    },
    onSuccess: () => {
      addToast({
        type: "success",
        title: "Huella enrolada",
        description: "Ahora puedes buscar coincidencias",
        duration: 4000,
      });
      setShowEnrollPicker(false);
      setEnrollPersonId("");
      setStep("analyze");
    },
    onError: (err: Error) => {
      addToast({ type: "error", title: "Error al enrolar", description: err.message });
    },
  });

  const createCaseMutation = useMutation({
    mutationFn: (input: CaseCreateInput) => createCase(input),
    onSuccess: (newCase) => {
      addToast({
        type: "success",
        title: "Caso creado",
        description: `${newCase.case_number} — ${newCase.title}`,
        duration: 4000,
      });
      setShowCreateCase(false);
      setCaseNumber("");
      setCaseTitle("");
      navigate(`/cases/${newCase.id}/compare`);
    },
    onError: (err: Error) => {
      addToast({ type: "error", title: "Error al crear caso", description: err.message });
    },
  });

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (!VALID_TYPES.includes(file.type as (typeof VALID_TYPES)[number])) {
        addToast({
          type: "error",
          title: "Tipo inválido",
          description: "BMP, PNG o JPEG",
        });
        return;
      }
      if (file.size > MAX_BYTES) {
        addToast({
          type: "error",
          title: "Archivo demasiado grande",
          description: "Máx 10MB",
        });
        return;
      }
      const reader = new FileReader();
      reader.onload = (ev) => {
        setLatentPreview(ev.target?.result as string);
        setLatentFile(file);
        setStep("upload");
        setPreview(null);
        setSearchResult(null);
        setSelectedCandidate(null);
      };
      reader.readAsDataURL(file);
    },
    [addToast],
  );

  const handleAnalyze = useCallback(() => {
    if (!latentFile) return;
    previewMutation.mutate(latentFile);
  }, [latentFile, previewMutation]);

  const handleSearch = useCallback(() => {
    if (!latentFile) return;
    searchMutation.mutate(latentFile);
  }, [latentFile, searchMutation]);

  const handleEnroll = useCallback(() => {
    if (!latentFile || !enrollPersonId) return;
    setEnrolling(true);
    enrollMutation.mutate(
      { personId: enrollPersonId, file: latentFile },
      { onSettled: () => setEnrolling(false) },
    );
  }, [latentFile, enrollPersonId, enrollMutation]);

  const handleOpenCreateCase = useCallback((candidate: MatchCandidate) => {
    const label = candidate.full_name ?? candidate.external_id ?? candidate.person_id.slice(0, 8);
    const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    setCaseNumber(`LATENT-${stamp}-${candidate.person_id.slice(0, 4).toUpperCase()}`);
    setCaseTitle(`Identificación latente vs ${label}`);
    setShowCreateCase(true);
  }, []);

  const handleSubmitCreateCase = useCallback(() => {
    if (!selectedCandidate) return;
    if (!caseNumber.trim() || !caseTitle.trim()) {
      addToast({ type: "error", title: "Campos incompletos" });
      return;
    }
    createCaseMutation.mutate({
      case_number: caseNumber.trim(),
      title: caseTitle.trim(),
      description: `Generado desde análisis top-level. Candidato: ${selectedCandidate.full_name ?? selectedCandidate.external_id ?? selectedCandidate.person_id}.`,
      status: "open",
    });
  }, [selectedCandidate, caseNumber, caseTitle, addToast, createCaseMutation]);

  const handleReset = useCallback(() => {
    setLatentFile(null);
    setLatentPreview(null);
    setPreview(null);
    setSearchResult(null);
    setSelectedCandidate(null);
    setStep("upload");
    setShowEnrollPicker(false);
    setShowCreateCase(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const probeMinutiae: MinutiaPoint[] = (searchResult?.probe_minutiae ?? []).map(
    (m) => ({ x: m.x, y: m.y, angle: m.angle, type: m.type }),
  );

  return (
    <div className="min-h-screen bg-background text-foreground p-8 font-sans dark">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex items-center justify-between border-b border-border pb-6">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/")}
              aria-label="Volver al panel"
            >
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div className="p-2 bg-primary/10 rounded-full">
              <Fingerprint className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Análisis de Huella</h1>
              <p className="text-muted-foreground text-sm">
                Subí, analizá, buscá — todo en un solo paso
              </p>
            </div>
          </div>
          {latentFile && (
            <Button variant="ghost" onClick={handleReset}>
              Empezar de nuevo
            </Button>
          )}
        </header>

        {/* Step 1: Upload */}
        {!latentFile && (
          <Card
            className="border-2 border-dashed border-border hover:border-primary/50 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <CardContent className="flex flex-col items-center justify-center py-20">
              <Upload className="w-16 h-16 mb-4 text-muted-foreground opacity-50" />
              <h2 className="text-xl font-semibold mb-2">Subí la huella latente</h2>
              <p className="text-sm text-muted-foreground max-w-md text-center">
                Una foto del levantado en la escena. BMP, PNG o JPEG. Máximo 10MB.
              </p>
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleFileChange}
              />
            </CardContent>
          </Card>
        )}

        {/* After upload: show image + actions */}
        {latentFile && latentPreview && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left: image + extract */}
              <Card className="border-border/60">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    <Fingerprint className="w-4 h-4" />
                    Huella cargada
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="bg-muted/20 rounded-lg overflow-hidden border border-border flex items-center justify-center min-h-[280px]">
                    <img
                      src={latentPreview}
                      alt="Huella"
                      className="max-w-full max-h-[360px] object-contain"
                    />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="w-3.5 h-3.5 mr-1.5" />
                      Cambiar
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleAnalyze}
                      disabled={previewMutation.isPending}
                      className="flex-1"
                    >
                      {previewMutation.isPending ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                          Procesando…
                        </>
                      ) : (
                        <>
                          <Fingerprint className="w-3.5 h-3.5 mr-1.5" />
                          Extraer minucias
                        </>
                      )}
                    </Button>
                  </div>
                  <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                </CardContent>
              </Card>

              {/* Center: extracted minutiae preview */}
              <Card className="border-border/60">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
                    Resultado
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {!preview ? (
                    <div className="flex flex-col items-center justify-center min-h-[200px] text-muted-foreground text-sm">
                      Presioná "Extraer minucias" para analizar
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="text-center">
                        <div className="text-4xl font-bold text-primary">
                          {preview.minutiae.length}
                        </div>
                        <div className="text-xs uppercase tracking-wider text-muted-foreground">
                          minucias detectadas
                        </div>
                      </div>
                      {preview.processed_image && (
                        <img
                          src={`data:image/png;base64,${preview.processed_image}`}
                          alt="Gabor enhanced"
                          className="w-full rounded border border-border"
                        />
                      )}
                      <Button
                        size="sm"
                        onClick={handleSearch}
                        disabled={preview.minutiae.length === 0 || searchMutation.isPending}
                        className="w-full"
                      >
                        {searchMutation.isPending ? (
                          <>
                            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                            Buscando…
                          </>
                        ) : (
                          <>
                            <Search className="w-3.5 h-3.5 mr-1.5" />
                            Buscar en la base
                          </>
                        )}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Right: action picker */}
              <Card className="border-border/60">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
                    ¿Qué querés hacer?
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {searchResult && searchResult.candidates.length === 0 && (
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => setShowEnrollPicker(true)}
                      className="w-full"
                    >
                      <UserPlus className="w-3.5 h-3.5 mr-1.5" />
                      No hay match — enrolar
                    </Button>
                  )}
                  {showEnrollPicker ? (
                    <PersonPicker
                      onSelect={setEnrollPersonId}
                      onCancel={() => setShowEnrollPicker(false)}
                      onConfirm={handleEnroll}
                      selectedId={enrollPersonId}
                      loading={enrolling}
                    />
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowEnrollPicker(true)}
                      className="w-full"
                    >
                      <UserPlus className="w-3.5 h-3.5 mr-1.5" />
                      Enrolar como nueva captura
                    </Button>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Step: results */}
            {searchResult && (
              <Card className="border-border/60">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Search className="w-4 h-4" />
                    Resultados ({searchResult.candidates.length})
                  </CardTitle>
                  <CardDescription>
                    {searchResult.candidates.length > 0
                      ? "Ordenados por similitud. Click en uno para ver detalle."
                      : "Ningún candidato superó el umbral."}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {searchResult.candidates.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <XCircle className="w-10 h-10 mx-auto mb-2 opacity-40" />
                      <p className="text-sm">Sin coincidencias. Probá enrolar.</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {searchResult.candidates.map((candidate, index) => (
                        <div key={candidate.person_id} className="space-y-2">
                          <CandidateCard
                            candidate={candidate}
                            rank={index + 1}
                            isSelected={selectedCandidate?.person_id === candidate.person_id}
                            onSelect={() => setSelectedCandidate(candidate)}
                          />
                          {selectedCandidate?.person_id === candidate.person_id && (
                            <div className="flex gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleOpenCreateCase(candidate)}
                                className="flex-1"
                              >
                                <FilePlus className="w-3.5 h-3.5 mr-1.5" />
                                Crear caso
                              </Button>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Selected candidate detail */}
            {selectedCandidate && searchResult && (
              <CandidateDetailPanel
                candidate={selectedCandidate}
                probeImageUrl={latentPreview}
                probeMinutiae={probeMinutiae}
                candidateImageUrl={null}
                onDismiss={() => setSelectedCandidate(null)}
              />
            )}
          </>
        )}

        {/* Create-case modal */}
        {showCreateCase && selectedCandidate && (
          <div
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
            onClick={() => !createCaseMutation.isPending && setShowCreateCase(false)}
          >
            <Card
              className="w-full max-w-md border-border/60"
              onClick={(e) => e.stopPropagation()}
            >
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <FilePlus className="w-4 h-4" />
                  Crear caso
                </CardTitle>
                <CardDescription>
                  Candidato: {selectedCandidate.full_name ?? selectedCandidate.external_id}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <label className="text-xs font-medium text-muted-foreground">
                    Número de caso
                  </label>
                  <input
                    value={caseNumber}
                    onChange={(e) => setCaseNumber(e.target.value)}
                    className="w-full mt-1 px-3 py-2 bg-background border border-border rounded text-sm"
                    maxLength={50}
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-foreground">
                    Título
                  </label>
                  <input
                    value={caseTitle}
                    onChange={(e) => setCaseTitle(e.target.value)}
                    className="w-full mt-1 px-3 py-2 bg-background border border-border rounded text-sm"
                    maxLength={300}
                  />
                </div>
                <div className="flex gap-2 justify-end pt-2">
                  <Button
                    variant="ghost"
                    onClick={() => setShowCreateCase(false)}
                    disabled={createCaseMutation.isPending}
                  >
                    Cancelar
                  </Button>
                  <Button
                    onClick={handleSubmitCreateCase}
                    disabled={createCaseMutation.isPending}
                  >
                    {createCaseMutation.isPending ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                        Creando…
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                        Crear
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        <div className="text-xs text-muted-foreground text-center pt-4">
          Paso actual: {step}
        </div>
      </div>
    </div>
  );
}

interface PersonPickerProps {
  selectedId: string;
  onSelect: (id: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
  loading: boolean;
}

function PersonPicker({ selectedId, onSelect, onCancel, onConfirm, loading }: PersonPickerProps) {
  const { data: persons, isLoading } = useQuery({
    queryKey: ["persons"],
    queryFn: () => listPersons(0, 100),
  });
  return (
    <div className="space-y-2 p-3 bg-muted/30 rounded-lg border border-border">
      <label className="text-xs font-medium text-muted-foreground">
        Persona
      </label>
      <select
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full px-2 py-1.5 bg-background border border-border rounded text-sm"
        disabled={isLoading}
      >
        <option value="">{isLoading ? "Cargando…" : "— seleccionar —"}</option>
        {persons?.map((p) => (
          <option key={p.id} value={p.id}>
            {p.full_name ?? p.external_id}
          </option>
        ))}
      </select>
      <div className="flex gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Cancelar
        </Button>
        <Button
          size="sm"
          onClick={onConfirm}
          disabled={!selectedId || loading}
        >
          {loading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              Enrolando…
            </>
          ) : (
            "Enrolar"
          )}
        </Button>
      </div>
    </div>
  );
}
