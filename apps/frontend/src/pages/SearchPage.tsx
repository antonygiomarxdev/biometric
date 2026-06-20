import { useState, useRef, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Search,
  Upload,
  Fingerprint,
  Loader2,
  XCircle,
  FilePlus,
  ArrowLeft,
  CheckCircle2,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/toast";
import { CandidateCard } from "@/components/fingerprint/CandidateCard";
import { CandidateDetailPanel } from "@/components/fingerprint/CandidateDetailPanel";
import {
  searchMatching,
  createCase,
  getMinutiaeForImage,
  type MatchCandidate,
  type MatchSearchResponse,
  type MinutiaPoint,
  type CaseCreateInput,
} from "@/lib/api";

const VALID_TYPES = ["image/bmp", "image/png", "image/jpeg", "image/jpg"];
const MAX_BYTES = 10 * 1024 * 1024;

interface CreateCaseModalState {
  open: boolean;
  candidate: MatchCandidate | null;
}

export default function SearchPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const [latentFile, setLatentFile] = useState<File | null>(null);
  const [latentPreview, setLatentPreview] = useState<string | null>(null);
  const [probePreviewUrl, setProbePreviewUrl] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<MatchSearchResponse | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<MatchCandidate | null>(null);
  const [createModal, setCreateModal] = useState<CreateCaseModalState>({
    open: false,
    candidate: null,
  });
  const [caseNumber, setCaseNumber] = useState("");
  const [caseTitle, setCaseTitle] = useState("");

  const searchMutation = useMutation({
    mutationFn: (file: File) => searchMatching(file, 10),
    onSuccess: (result) => {
      setSearchResult(result);
      setSelectedCandidate(result.candidates[0] ?? null);
      addToast({
        type: "success",
        title: "Búsqueda completada",
        description: `${result.candidates.length} candidato${result.candidates.length !== 1 ? "s" : ""} encontrado${result.candidates.length !== 1 ? "s" : ""}`,
        duration: 3000,
      });
    },
    onError: (error: unknown) => {
      addToast({
        type: "error",
        title: "Error en la búsqueda",
        description: error instanceof Error ? error.message : "Error desconocido",
      });
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
      queryClient.invalidateQueries({ queryKey: ["cases"] });
      setCreateModal({ open: false, candidate: null });
      setCaseNumber("");
      setCaseTitle("");
      navigate(`/cases/${newCase.id}/compare`);
    },
    onError: (error: unknown) => {
      addToast({
        type: "error",
        title: "Error al crear caso",
        description: error instanceof Error ? error.message : "Error desconocido",
      });
    },
  });

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      if (!VALID_TYPES.includes(file.type)) {
        addToast({
          type: "error",
          title: "Tipo de archivo inválido",
          description: "Selecciona una imagen BMP, PNG o JPEG",
        });
        return;
      }

      if (file.size > MAX_BYTES) {
        addToast({
          type: "error",
          title: "Archivo demasiado grande",
          description: "El archivo no debe exceder 10MB",
        });
        return;
      }

      const reader = new FileReader();
      reader.onload = (ev) => {
        setLatentPreview(ev.target?.result as string);
        setLatentFile(file);
        setSearchResult(null);
        setSelectedCandidate(null);
        setProbePreviewUrl(null);
      };
      reader.readAsDataURL(file);

      getMinutiaeForImage(file)
        .then((res) => setProbePreviewUrl(res.processed_image_url))
        .catch(() => {});
    },
    [addToast],
  );

  const handleSearch = useCallback(() => {
    if (!latentFile) return;
    searchMutation.mutate(latentFile);
  }, [latentFile, searchMutation]);

  const handleReset = useCallback(() => {
    setLatentFile(null);
    setLatentPreview(null);
    setSearchResult(null);
    setSelectedCandidate(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const handleOpenCreateCase = useCallback(
    (candidate: MatchCandidate) => {
      const label = candidate.full_name ?? candidate.external_id ?? candidate.person_id.slice(0, 8);
      const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
      setCaseNumber(`LATENT-${stamp}-${candidate.person_id.slice(0, 4).toUpperCase()}`);
      setCaseTitle(`Identificación latente vs ${label}`);
      setCreateModal({ open: true, candidate });
    },
    [],
  );

  const handleSubmitCreateCase = useCallback(() => {
    if (!createModal.candidate) return;
    if (!caseNumber.trim() || !caseTitle.trim()) {
      addToast({
        type: "error",
        title: "Campos incompletos",
        description: "Número y título son obligatorios",
      });
      return;
    }
    createCaseMutation.mutate({
      case_number: caseNumber.trim(),
      title: caseTitle.trim(),
      description: `Generado desde búsqueda top-level. Candidato: ${createModal.candidate.full_name ?? createModal.candidate.external_id ?? createModal.candidate.person_id}.`,
      status: "open",
    });
  }, [createModal.candidate, caseNumber, caseTitle, addToast, createCaseMutation]);

  useEffect(() => {
    return () => {
      if (latentPreview?.startsWith("blob:") || latentPreview?.startsWith("data:")) {
        // FileReader data URIs are cleaned up by browser tab close; no-op
      }
    };
  }, [latentPreview]);

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
              <Search className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Búsqueda Rápida de Huella
              </h1>
              <p className="text-muted-foreground text-sm">
                Identificación top-level contra toda la base enrolada
              </p>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="border-border/60">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Fingerprint className="w-4 h-4" />
                Huella Latente
              </CardTitle>
              <CardDescription>
                Sube la huella levantada en la escena para identificar al individuo
              </CardDescription>
            </CardHeader>
            <CardContent>
              {latentPreview ? (
                <div className="space-y-4">
                  <div className="bg-muted/20 rounded-lg overflow-hidden border border-border flex items-center justify-center min-h-[300px]">
                    <img
                      src={latentPreview}
                      alt="Huella latente"
                      className="max-w-full max-h-[400px] object-contain"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="w-3.5 h-3.5 mr-1.5" />
                      Cambiar imagen
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleSearch}
                      disabled={searchMutation.isPending || !latentFile}
                    >
                      {searchMutation.isPending ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                          Buscando...
                        </>
                      ) : (
                        <>
                          <Search className="w-3.5 h-3.5 mr-1.5" />
                          Buscar coincidencias
                        </>
                      )}
                    </Button>
                    {searchResult && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleReset}
                      >
                        <XCircle className="w-3.5 h-3.5 mr-1.5" />
                        Limpiar
                      </Button>
                    )}
                  </div>
                </div>
              ) : (
                <div
                  className="flex flex-col items-center justify-center h-full min-h-[300px] text-muted-foreground border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary/50 transition-colors"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-12 h-12 mb-3 opacity-40" />
                  <p className="text-sm font-medium">Subir huella latente</p>
                  <p className="text-xs mt-1">BMP, PNG, JPEG — máx 10MB</p>
                </div>
              )}

              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleFileChange}
              />
            </CardContent>
          </Card>

          <Card className="border-border/60 h-full">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Fingerprint className="w-4 h-4" />
                Candidatos
              </CardTitle>
              <CardDescription>
                {searchResult
                  ? `${searchResult.candidates.length} posible${searchResult.candidates.length !== 1 ? "s" : ""} coincidencia${searchResult.candidates.length !== 1 ? "s" : ""}`
                  : "Sube una huella y presiona buscar"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {searchMutation.isPending ? (
                <div className="flex flex-col items-center justify-center min-h-[300px] text-muted-foreground">
                  <Loader2 className="w-8 h-8 animate-spin mb-3" />
                  <p className="text-sm">Buscando en la base AFIS...</p>
                </div>
              ) : !searchResult ? (
                <div className="flex flex-col items-center justify-center min-h-[300px] text-muted-foreground">
                  <Search className="w-12 h-12 mb-3 opacity-20" />
                  <p className="text-sm">
                    {latentPreview
                      ? "Presiona 'Buscar coincidencias' para comenzar"
                      : "Sube una huella latente"}
                  </p>
                </div>
              ) : searchResult.candidates.length === 0 ? (
                <div className="flex flex-col items-center justify-center min-h-[300px] text-muted-foreground">
                  <XCircle className="w-10 h-10 mb-3 opacity-40" />
                  <p className="text-sm font-medium">Sin coincidencias</p>
                  <p className="text-xs mt-1">
                    Ningún candidato superó el umbral de similitud.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {searchResult.candidates.map((candidate, index) => (
                    <div key={candidate.person_id} className="space-y-1">
                      <CandidateCard
                        candidate={candidate}
                        rank={index + 1}
                        isSelected={selectedCandidate?.person_id === candidate.person_id}
                        onSelect={() => setSelectedCandidate(candidate)}
                      />
                      {selectedCandidate?.person_id === candidate.person_id && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full"
                          onClick={() => handleOpenCreateCase(candidate)}
                        >
                          <FilePlus className="w-3.5 h-3.5 mr-1.5" />
                          Crear caso con este candidato
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {selectedCandidate && searchResult && (
          <CandidateDetailPanel
            candidate={selectedCandidate}
            probeImageUrl={searchResult.probe_image_url}
            probeMinutiae={probeMinutiae}
            onDismiss={() => setSelectedCandidate(null)}
          />
        )}

        {createModal.open && createModal.candidate && (
          <div
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
            onClick={() => !createCaseMutation.isPending && setCreateModal({ open: false, candidate: null })}
          >
            <Card
              className="w-full max-w-md border-border/60"
              onClick={(e) => e.stopPropagation()}
            >
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <FilePlus className="w-4 h-4" />
                  Crear caso desde búsqueda
                </CardTitle>
                <CardDescription>
                  Candidato: {createModal.candidate.full_name ?? createModal.candidate.external_id ?? createModal.candidate.person_id.slice(0, 8)}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">
                    Número de caso
                  </label>
                  <Input
                    value={caseNumber}
                    onChange={(e) => setCaseNumber(e.target.value)}
                    placeholder="CASO-2026-001"
                    maxLength={50}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">
                    Título
                  </label>
                  <Input
                    value={caseTitle}
                    onChange={(e) => setCaseTitle(e.target.value)}
                    placeholder="Identificación latente vs candidato"
                    maxLength={300}
                  />
                </div>
                <div className="flex gap-2 justify-end pt-2">
                  <Button
                    variant="ghost"
                    onClick={() => setCreateModal({ open: false, candidate: null })}
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
                        Creando...
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                        Crear caso
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
