import {
  WorkflowStepper,
  UploadDropzone,
} from "@/components/analisis/WorkflowStepper";
import { ProbePanel } from "@/components/analisis/ProbePanel";
import { ResultsPanel } from "@/components/analisis/ResultsPanel";
import { useProbeProcessor } from "@/hooks/analisis/useProbeProcessor";
import { useSearchManager } from "@/hooks/analisis/useSearchManager";
import { Button } from "@/components/ui/button";
import {
  ArrowLeft,
  Fingerprint,
  Loader2,
  Search,
  UserPlus,
  FilePlus,
  CheckCircle2,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useCallback, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/toast";
import {
  createCase,
  createFingerprintSlot,
  enrollFingerprint,
  listPersons,
  type CaseCreateInput,
  type MatchCandidate,
} from "@/lib/api";
import { useMutation, useQuery } from "@tanstack/react-query";

export default function AnalisisPage() {
  const navigate = useNavigate();
  const { addToast } = useToast();
  const {
    latentFile,
    probeDataUrl,
    probePreviewUrl,
    probeMinutiae,
    isPreviewLoading,
    handleFile,
    reset: resetProbe,
  } = useProbeProcessor();

  const { searchResult, searchMutation, reset: resetSearch } = useSearchManager();

  const [showEnrollPicker, setShowEnrollPicker] = useState(false);
  const [enrollPersonId, setEnrollPersonId] = useState("");
  const [enrolling, setEnrolling] = useState(false);
  const [showCreateCase, setShowCreateCase] = useState(false);
  const [caseNumber, setCaseNumber] = useState("");
  const [caseTitle, setCaseTitle] = useState("");

  const enrollMutation = useMutation({
    mutationFn: async ({ personId, file }: { personId: string; file: File }) => {
      const slot = await createFingerprintSlot(personId, 0, "rolled");
      return enrollFingerprint(slot.id, file);
    },
    onSuccess: () => {
      addToast({
        type: "success",
        title: "Huella enrolada",
        description: "Ahora podés buscar coincidencias.",
        duration: 4000,
      });
      setShowEnrollPicker(false);
      setEnrollPersonId("");
    },
    onError: (err: Error) => {
      addToast({
        type: "error",
        title: "Error al enrolar",
        description: err.message,
      });
    },
  });

  const createCaseMutation = useMutation({
    mutationFn: (input: CaseCreateInput) => createCase(input),
    onSuccess: (newCase) => {
      addToast({
        type: "success",
        title: "Caso creado",
        description: `${newCase.case_number}`,
        duration: 4000,
      });
      setShowCreateCase(false);
      navigate(`/cases/${newCase.id}/compare`);
    },
    onError: (err: Error) => {
      addToast({
        type: "error",
        title: "Error al crear caso",
        description: err.message,
      });
    },
  });

  const { data: persons } = useQuery({
    queryKey: ["persons"],
    queryFn: () => listPersons(0, 100),
  });

  const handleSearch = useCallback(() => {
    if (!latentFile) return;
    searchMutation.mutate(latentFile);
  }, [latentFile, searchMutation]);

  const handleEnroll = useCallback(() => {
    if (!latentFile || !enrollPersonId) return;
    setEnrolling(true);
    enrollMutation.mutate(
      { personId: enrollPersonId, file: latentFile },
      { onSettled: () => setEnrolling(false) }
    );
  }, [latentFile, enrollPersonId, enrollMutation]);

  const handleOpenCreateCase = useCallback((c: MatchCandidate) => {
    const label = c.full_name ?? c.external_id ?? c.person_id.slice(0, 8);
    const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    setCaseNumber(`LATENT-${stamp}-${c.person_id.slice(0, 4).toUpperCase()}`);
    setCaseTitle(`Identificación latente vs ${label}`);
    setShowCreateCase(true);
  }, []);

  const handleReset = useCallback(() => {
    resetProbe();
    resetSearch();
  }, [resetProbe, resetSearch]);

  const currentStep: 0 | 1 | 2 = (() => {
    if (searchResult) return 2;
    if (probeDataUrl) return 1;
    return 0;
  })();

  return (
    <div className="min-h-screen bg-background text-foreground p-6 font-sans dark">
      <div className="max-w-[1400px] mx-auto space-y-4">
        <header className="flex items-center justify-between border-b border-border pb-3">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate("/")}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <Fingerprint className="w-5 h-5 text-primary" />
            <h1 className="text-xl font-bold tracking-tight">
              Análisis de Huella
            </h1>
          </div>
          {latentFile && (
            <Button variant="ghost" onClick={handleReset}>
              Empezar de nuevo
            </Button>
          )}
        </header>

        <WorkflowStepper
          current={currentStep}
          searchRunning={searchMutation.isPending}
          minutiae={
            searchResult?.probe_minutiae.length ?? probeMinutiae.length ?? 0
          }
          candidateCount={searchResult?.candidates.length ?? 0}
          queryTimeMs={searchResult?.query_time_ms}
        >
          {!probeDataUrl ? (
            <UploadDropzone onFile={handleFile} running={isPreviewLoading} />
          ) : (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2 p-3 bg-card border border-border rounded-lg">
                <Button
                  size="sm"
                  onClick={handleSearch}
                  disabled={!latentFile || searchMutation.isPending}
                  className="flex-1 sm:flex-none"
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

                {searchResult && searchResult.candidates.length === 0 && (
                  <Button
                    size="sm"
                    variant="default"
                    onClick={() => setShowEnrollPicker(!showEnrollPicker)}
                    className="flex-1 sm:flex-none"
                  >
                    <UserPlus className="w-3.5 h-3.5 mr-1.5" />
                    No hay match — enrolar
                  </Button>
                )}

                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setShowEnrollPicker(!showEnrollPicker)}
                >
                  <UserPlus className="w-3.5 h-3.5 mr-1.5" />
                  Enrolar
                </Button>

                <div className="ml-auto text-xs text-muted-foreground font-mono">
                  {searchResult
                    ? `${searchResult.probe_minutiae.length} minucias · ${searchResult.candidates.length} candidatos · ${searchResult.query_time_ms}ms`
                    : probeMinutiae.length > 0
                    ? `${probeMinutiae.length} minucias`
                    : isPreviewLoading
                    ? "procesando..."
                    : "—"}
                </div>
              </div>

              {showEnrollPicker && (
                <Card className="border-border/60">
                  <CardContent className="p-3 flex flex-wrap items-center gap-2">
                    <span className="text-sm text-muted-foreground">
                      Persona:
                    </span>
                    <select
                      value={enrollPersonId}
                      onChange={(e) => setEnrollPersonId(e.target.value)}
                      className="flex-1 min-w-[200px] px-2 py-1.5 bg-background border border-border rounded text-sm"
                    >
                      <option value="">— seleccionar —</option>
                      {persons?.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.full_name ?? p.external_id}
                        </option>
                      ))}
                    </select>
                    <Button
                      size="sm"
                      onClick={handleEnroll}
                      disabled={!enrollPersonId || enrolling}
                    >
                      {enrolling ? (
                        <>
                          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                          Enrolando…
                        </>
                      ) : (
                        "Enrolar"
                      )}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setShowEnrollPicker(false)}
                    >
                      Cancelar
                    </Button>
                  </CardContent>
                </Card>
              )}

              <ProbePanel
                probeDataUrl={probeDataUrl}
                probePreviewUrl={probePreviewUrl}
                probeMinutiae={probeMinutiae}
                searchResult={searchResult}
                selectedCandidate={searchResult?.candidates[0]}
                onFile={handleFile}
                isLoading={isPreviewLoading}
              />

              <ResultsPanel
                searchResult={searchResult}
                probeImageUrl={
                  searchResult?.probe_image_url ?? probePreviewUrl ?? probeDataUrl
                }
                probeMinutiae={searchResult?.probe_minutiae ?? []}
              />
            </div>
          )}
        </WorkflowStepper>
      </div>
    </div>
  );
}
