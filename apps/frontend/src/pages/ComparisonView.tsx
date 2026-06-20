import { useState, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Fingerprint,
  Upload,
  Loader2,
  ArrowLeft,
  CheckCircle,
  XCircle,
  HelpCircle,
  AlertCircle,
  Star,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/toast";
import {
  getCase,
  listEvidence,
  searchMatching,
  createDecision,
} from "@/lib/api";
import type { EvidenceResponse, MatchCandidate, MinutiaPoint } from "@/lib/api";
import { CandidateCard } from "@/components/fingerprint/CandidateCard";
import { CandidateDetailPanel } from "@/components/fingerprint/CandidateDetailPanel";

type Verdict = "Identificación" | "Exclusión" | "Inconcluso";

const VERDICT_BUTTONS: {
  verdict: Verdict;
  label: string;
  icon: typeof CheckCircle;
  variant: "default" | "destructive" | "secondary";
}[] = [
  {
    verdict: "Identificación",
    label: "Identificación",
    icon: CheckCircle,
    variant: "default",
  },
  {
    verdict: "Exclusión",
    label: "Exclusión",
    icon: XCircle,
    variant: "destructive",
  },
  {
    verdict: "Inconcluso",
    label: "Inconcluso",
    icon: HelpCircle,
    variant: "secondary",
  },
];

function LoadingPanel({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-muted-foreground">
      <Loader2 className="w-8 h-8 animate-spin mb-3" />
      <p className="text-sm">{label}</p>
    </div>
  );
}

function EmptyLatentPanel({ onUpload }: { onUpload: () => void }) {
  return (
    <div
      className="flex flex-col items-center justify-center h-full min-h-[400px] text-muted-foreground border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary/50 transition-colors"
      onClick={onUpload}
    >
      <Upload className="w-12 h-12 mb-3 opacity-40" />
      <p className="text-sm font-medium">Subir huella latente</p>
      <p className="text-xs mt-1">BMP, PNG, JPEG</p>
    </div>
  );
}

export default function ComparisonView() {
  const { caseId } = useParams<{ caseId: string }>();
  const navigate = useNavigate();
  const { addToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // State
  const [latentPreview, setLatentPreview] = useState<string | null>(null);
  const [latentFile, setLatentFile] = useState<File | null>(null);
  const [candidates, setCandidates] = useState<MatchCandidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<MatchCandidate | null>(null);
  const [probeMinutiae, setProbeMinutiae] = useState<MinutiaPoint[]>([]);
  const [probeImageUrl, setProbeImageUrl] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submittedVerdict, setSubmittedVerdict] = useState<Verdict | null>(null);

  // Fetch case details
  const {
    data: caseData,
    isLoading: caseLoading,
    isError: caseError,
  } = useQuery({
    queryKey: ["case", caseId],
    queryFn: () => getCase(caseId!),
    enabled: caseId !== undefined,
  });

  // Fetch evidence for the case
  const {
    data: evidenceList,
  } = useQuery({
    queryKey: ["evidencias", caseId],
    queryFn: () => listEvidence(caseId),
    enabled: caseId !== undefined,
  });

  const handleFileUpload = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const validTypes = ["image/bmp", "image/png", "image/jpeg", "image/jpg"];
      if (!validTypes.includes(file.type)) {
        addToast({
          type: "error",
          title: "Tipo de archivo inválido",
          description: "Selecciona una imagen BMP, PNG o JPEG",
        });
        return;
      }

      if (file.size > 10 * 1024 * 1024) {
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
        setCandidates([]);
        setSelectedCandidate(null);
        setProbeMinutiae([]);
        setSubmittedVerdict(null);
      };
      reader.readAsDataURL(file);
    },
    [addToast],
  );

  const handleSearch = useCallback(async () => {
    if (!latentFile || !caseId) return;

    setIsSearching(true);
    setCandidates([]);
    setSelectedCandidate(null);
    setSubmittedVerdict(null);

    try {
      addToast({
        type: "info",
        title: "Buscando coincidencias...",
        description: "Procesando huella latente contra base de datos AFIS",
        duration: 3000,
      });

      const result = await searchMatching(latentFile);

      setProbeImageUrl(result.probe_image_url);
      if (result.candidates.length > 0) {
        setCandidates(result.candidates);
        setProbeMinutiae(result.probe_minutiae);
        setSelectedCandidate(result.candidates[0]);
        addToast({
          type: "success",
          title: "Búsqueda completada",
          description: `${result.total_candidates} candidato${result.total_candidates !== 1 ? "s" : ""} encontrado${result.total_candidates !== 1 ? "s" : ""}`,
          duration: 3000,
        });
      } else {
        setProbeMinutiae(result.probe_minutiae);
        addToast({
          type: "info",
          title: "Sin coincidencias",
          description: "No se encontraron candidatos que superen el umbral",
          duration: 4000,
        });
      }
    } catch (error) {
      addToast({
        type: "error",
        title: "Error en la búsqueda",
        description:
          error instanceof Error
            ? error.message
            : "No se pudo completar la búsqueda",
      });
    } finally {
      setIsSearching(false);
    }
  }, [latentFile, caseId, addToast]);

  const handleDecision = useCallback(
    async (verdict: Verdict) => {
      if (!caseId || !selectedCandidate) return;

      setIsSubmitting(true);
      try {
        const evidenceId = evidenceList?.items[0]?.id ?? null;

        await createDecision({
          case_id: caseId,
          evidence_id: evidenceId,
          verdict,
          comments: null,
        });

        setSubmittedVerdict(verdict);
        addToast({
          type: "success",
          title: "Decisión registrada",
          description: `Veredicto: ${verdict} — Registrado en cadena de custodia`,
          duration: 5000,
        });
      } catch (error) {
        addToast({
          type: "error",
          title: "Error al registrar decisión",
          description:
            error instanceof Error
              ? error.message
              : "No se pudo registrar la decisión",
        });
      } finally {
        setIsSubmitting(false);
      }
    },
    [caseId, selectedCandidate, evidenceList, addToast],
  );

  // Upload a latent fingerprint from one of the case's existing evidence
  const handleUseEvidence = useCallback(
    (evidence: EvidenceResponse) => {
      if (evidence.image_path) {
        const API_BASE = import.meta.env.VITE_API_URL ?? "";
        setLatentPreview(
          `${API_BASE}/api/v1/evidence/${evidence.id}/image`,
        );
        setLatentFile(null);
        setCandidates([]);
        setSelectedCandidate(null);
        setProbeMinutiae([]);
        setSubmittedVerdict(null);
      }
    },
    [],
  );

  if (caseLoading) {
    return (
      <div className="min-h-screen bg-background text-foreground p-8 font-sans dark flex items-center justify-center">
        <LoadingPanel label="Cargando caso..." />
      </div>
    );
  }

  if (caseError || !caseData) {
    return (
      <div className="min-h-screen bg-background text-foreground p-8 font-sans dark">
        <div className="max-w-5xl mx-auto">
          <Button variant="outline" onClick={() => navigate("/")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Volver al panel
          </Button>
          <Card className="mt-6 border-destructive/50 bg-destructive/5">
            <CardContent className="flex items-center gap-3 py-6">
              <AlertCircle className="w-5 h-5 text-destructive shrink-0" />
              <p className="text-destructive font-medium">
                No se pudo cargar el caso solicitado.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  const canSubmitDecision =
    candidates.length > 0 && selectedCandidate !== null && !isSubmitting && submittedVerdict === null;

  return (
    <div className="min-h-screen bg-background text-foreground p-8 font-sans dark">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Top bar */}
        <header className="flex items-center justify-between border-b border-border pb-6">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => navigate("/")}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-xl font-bold tracking-tight">
                  {caseData.title}
                </h1>
                <Badge
                  variant="outline"
                  className="text-xs font-medium capitalize"
                >
                  {caseData.status}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground font-mono">
                {caseData.case_number} — Comparación Forense
              </p>
            </div>
          </div>
        </header>

        {/* Latent fingerprint panel (full width) */}
        <section>
          <Card className="border-border/60">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Fingerprint className="w-4 h-4" />
                Huella Latente
              </CardTitle>
              <CardDescription>
                Evidencia recolectada en la escena del caso
              </CardDescription>
            </CardHeader>
            <CardContent>
              {latentPreview ? (
                <div className="space-y-4">
                  <div className="bg-muted/20 rounded-lg overflow-hidden border border-border flex items-center justify-center min-h-[350px]">
                    <img
                      src={latentPreview}
                      alt="Latent fingerprint preview"
                      className="max-w-full max-h-[450px] object-contain"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleFileUpload}
                    >
                      <Upload className="w-3.5 h-3.5 mr-1.5" />
                      Cambiar imagen
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleSearch}
                      disabled={isSearching || !latentFile}
                    >
                      {isSearching ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                          Buscando...
                        </>
                      ) : (
                        <>
                          <Fingerprint className="w-3.5 h-3.5 mr-1.5" />
                          Buscar coincidencias
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              ) : (
                <EmptyLatentPanel onUpload={handleFileUpload} />
              )}

              {/* Hidden file input */}
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleFileChange}
              />

              {/* Existing evidence gallery */}
              {evidenceList && evidenceList.items.length > 0 && !latentPreview && (
                <div className="mt-4 pt-4 border-t border-border">
                  <p className="text-xs text-muted-foreground mb-2">
                    Evidencia existente del caso:
                  </p>
                  <div className="space-y-1">
                    {evidenceList.items.map((ev) => (
                      <Button
                        key={ev.id}
                        variant="ghost"
                        size="sm"
                        className="w-full justify-start text-xs"
                        onClick={() => handleUseEvidence(ev)}
                      >
                        <Fingerprint className="w-3 h-3 mr-2" />
                        {ev.fingerprint_id}
                      </Button>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Candidate list (full width below latent panel) */}
        <section>
          <Card className="border-border/60">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Star className="w-4 h-4" />
                Candidatos (Top-K)
              </CardTitle>
              <CardDescription>
                {candidates.length > 0
                  ? `${candidates.length} posibles coincidencias ordenadas por similitud`
                  : "Sube una huella latente y busca coincidencias"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isSearching ? (
                <LoadingPanel label="Buscando en base de datos AFIS..." />
              ) : candidates.length > 0 ? (
                <div className="space-y-3">
                  {candidates.map((candidate, index) => (
                    <CandidateCard
                      key={candidate.person_id}
                      candidate={candidate}
                      rank={index + 1}
                      isSelected={selectedCandidate?.person_id === candidate.person_id}
                      onSelect={() => setSelectedCandidate(candidate)}
                    />
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full min-h-[200px] text-muted-foreground">
                  <Fingerprint className="w-12 h-12 mb-3 opacity-20" />
                  <p className="text-sm">
                    {latentPreview
                      ? "Presiona 'Buscar coincidencias' para comenzar"
                      : "Selecciona o sube una huella latente"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Candidate detail panel (full width, mounts when a card is selected) */}
        {selectedCandidate && (
          <CandidateDetailPanel
            candidate={selectedCandidate}
            probeImageUrl={probeImageUrl ?? latentPreview ?? ""}
            probeMinutiae={probeMinutiae}
            onDismiss={() => setSelectedCandidate(null)}
          />
        )}

        {/* Action buttons */}
        {selectedCandidate && (
          <Card className="border-border/60">
            <CardContent className="py-4">
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <p className="text-sm text-muted-foreground">
                    Candidato seleccionado:{" "}
                    <span className="font-semibold text-foreground">
                      {selectedCandidate.full_name ?? `Persona ${selectedCandidate.person_id.slice(0, 8)}`}
                    </span>
                    <span className="font-mono ml-1">
                      ({selectedCandidate.external_id ?? selectedCandidate.person_id})
                    </span>
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {submittedVerdict ? (
                    <Badge
                      variant="outline"
                      className="text-sm py-1.5 px-4 bg-primary/5 border-primary/30"
                    >
                      <CheckCircle className="w-4 h-4 mr-1.5 text-primary" />
                      Decisión registrada: {submittedVerdict}
                    </Badge>
                  ) : (
                    VERDICT_BUTTONS.map(({ verdict, label, icon: Icon }) => (
                      <Button
                        key={verdict}
                        variant={verdict === "Exclusión" ? "destructive" : verdict === "Inconcluso" ? "secondary" : "default"}
                        onClick={() => handleDecision(verdict)}
                        disabled={!canSubmitDecision}
                        className={verdict === "Identificación" ? "bg-green-600 hover:bg-green-700 text-white" : ""}
                      >
                        {isSubmitting ? (
                          <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
                        ) : (
                          <Icon className="w-4 h-4 mr-1.5" />
                        )}
                        {label}
                      </Button>
                    ))
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
