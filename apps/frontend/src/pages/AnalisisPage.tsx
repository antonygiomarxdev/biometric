import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Upload,
  Fingerprint,
  Loader2,
  CheckCircle2,
  ArrowLeft,
  UserPlus,
  FilePlus,
  Search,
  Trophy,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { CandidateDetailPanel } from "@/components/fingerprint/CandidateDetailPanel";
import { useToast } from "@/components/ui/toast";
import {
  getMinutiaeForImage,
  searchMatching,
  listPersons,
  createFingerprintSlot,
  enrollFingerprint,
  createCase,
  fetchCaptureImage,
  type MatchCandidate,
  type MatchSearchResponse,
  type MinutiaPoint,
  type FingerprintPreviewResponse,
  type CaseCreateInput,
} from "@/lib/api";

const VALID_TYPES = ["image/bmp", "image/png", "image/jpeg", "image/jpg"];
const MAX_BYTES = 10 * 1024 * 1024;

const PALETTE_HIT = "#ffffff";
const PALETTE_HIT_RING = "#22c55e";

// Tiered confidence thresholds for the perito. Above 0.9 is
// "high" (single best match, "OK"); 0.7-0.9 is "medium" (possible
// alternatives, requires review); below 0.7 is filtered server-side
// as noise. The actual server-side filter lives in
// `MCC_CONFIDENCE_THRESHOLD`.
const MATCH_THRESHOLD_GOOD = 0.9;
const MATCH_THRESHOLD_FAIR = 0.7;

export default function AnalisisPage() {
  const navigate = useNavigate();
  const { addToast } = useToast();

  const probeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const candidateCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const probeImgRef = useRef<HTMLImageElement | null>(null);
  const candidateImgRef = useRef<HTMLImageElement | null>(null);

  const [latentFile, setLatentFile] = useState<File | null>(null);
  const [probeDataUrl, setProbeDataUrl] = useState<string | null>(null);
  const [processedDataUrl, setProcessedDataUrl] = useState<string | null>(null);
  const [preview, setPreview] = useState<FingerprintPreviewResponse | null>(null);
  const [searchResult, setSearchResult] = useState<MatchSearchResponse | null>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);

  const [candidateDataUrl, setCandidateDataUrl] = useState<string | null>(null);
  const [candidateLoading, setCandidateLoading] = useState(false);

  const [showEnrollPicker, setShowEnrollPicker] = useState(false);
  const [enrollPersonId, setEnrollPersonId] = useState("");
  const [enrolling, setEnrolling] = useState(false);
  const [showCreateCase, setShowCreateCase] = useState(false);
  const [caseNumber, setCaseNumber] = useState("");
  const [caseTitle, setCaseTitle] = useState("");

  // ============================================================
  // Mutations / queries
  // ============================================================

  const previewMutation = useMutation({
    mutationFn: (file: File) => getMinutiaeForImage(file),
    onSuccess: (result) => {
      setPreview(result);
      if (result.processed_image) {
        setProcessedDataUrl(`data:image/png;base64,${result.processed_image}`);
      }
    },
    onError: (err: Error) => {
      addToast({ type: "error", title: "Error al procesar", description: err.message });
    },
  });

  const searchMutation = useMutation({
    mutationFn: (file: File) => searchMatching(file, 10),
    onSuccess: (result) => {
      setSearchResult(result);
      setSelectedIdx(0);
      if (result.candidates.length === 0) {
        addToast({
          type: "info",
          title: "Sin coincidencias",
          description: "Podés enrolar esta huella si pertenece a alguien nuevo.",
          duration: 5000,
        });
      } else {
        addToast({
          type: "success",
          title: `${result.candidates.length} candidato${result.candidates.length !== 1 ? "s" : ""} encontrado${result.candidates.length !== 1 ? "s" : ""}`,
          description: "Hacé click en uno para ver la comparación",
          duration: 4000,
        });
      }
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
        description: "Ahora podés buscar coincidencias.",
        duration: 4000,
      });
      setShowEnrollPicker(false);
      setEnrollPersonId("");
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
        description: `${newCase.case_number}`,
        duration: 4000,
      });
      setShowCreateCase(false);
      navigate(`/cases/${newCase.id}/compare`);
    },
    onError: (err: Error) => {
      addToast({ type: "error", title: "Error al crear caso", description: err.message });
    },
  });

  const { data: persons } = useQuery({
    queryKey: ["persons"],
    queryFn: () => listPersons(0, 100),
  });

  // ============================================================
  // Handlers
  // ============================================================

  const handleFile = useCallback(
    (file: File) => {
      if (!VALID_TYPES.includes(file.type as (typeof VALID_TYPES)[number])) {
        addToast({ type: "error", title: "Tipo inválido", description: "BMP, PNG o JPEG" });
        return;
      }
      if (file.size > MAX_BYTES) {
        addToast({ type: "error", title: "Archivo grande", description: "Máx 10MB" });
        return;
      }
      const reader = new FileReader();
      reader.onload = (ev) => {
        setProbeDataUrl(ev.target?.result as string);
        setLatentFile(file);
        setPreview(null);
        setSearchResult(null);
        setSelectedIdx(0);
        setCandidateDataUrl(null);
        setProcessedDataUrl(null);
        previewMutation.mutate(file);
      };
      reader.readAsDataURL(file);
    },
    [addToast, previewMutation],
  );

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

  const handleOpenCreateCase = useCallback((c: MatchCandidate) => {
    const label = c.full_name ?? c.external_id ?? c.person_id.slice(0, 8);
    const stamp = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    setCaseNumber(`LATENT-${stamp}-${c.person_id.slice(0, 4).toUpperCase()}`);
    setCaseTitle(`Identificación latente vs ${label}`);
    setShowCreateCase(true);
  }, []);

  const handleReset = useCallback(() => {
    setLatentFile(null);
    setProbeDataUrl(null);
    setProcessedDataUrl(null);
    setPreview(null);
    setSearchResult(null);
    setSelectedIdx(0);
    setCandidateDataUrl(null);
  }, []);

  // ============================================================
  // Load candidate image when selection changes
  // ============================================================

  const selectedCandidate: MatchCandidate | null =
    searchResult?.candidates[selectedIdx] ?? null;

  // Stepper state — drives the always-visible WorkflowStepper
  // 0=no image, 1=uploaded, 2=previewed, 3=searched (candidates available)
  const currentStep: 0 | 1 | 2 | 3 = (() => {
    if (searchResult) return 3;
    if (preview) return 2;
    if (probeDataUrl) return 1;
    return 0;
  })();

  useEffect(() => {
    if (!selectedCandidate) {
      setCandidateDataUrl(null);
      return;
    }
    const captureId = selectedCandidate.supporting_pairs[0]?.candidate_capture_id;
    if (!captureId) {
      setCandidateDataUrl(null);
      return;
    }
    let cancelled = false;
    setCandidateLoading(true);
    fetchCaptureImage(captureId)
      .then((url) => {
        if (!cancelled) {
          setCandidateDataUrl(url);
          setCandidateLoading(false);
        }
      })
      .catch((err: Error & { status?: number }) => {
        if (cancelled) return;
        setCandidateLoading(false);
        setCandidateDataUrl(null);
        // 503 = capture predates the enhanced_image column. Surface a
        // clear toast so the perito re-enrolls. There is no in-place
        // fallback — legacy data is treated as missing, not as something
        // to silently paper over.
        if (err.status === 503) {
          addToast({
            type: "warning",
            title: "Captura legacy sin imagen enhanced",
            description:
              "Esta captura se enroló antes de la migración. Re-enrolá para ver la comparación visual.",
            duration: 8000,
          });
        } else {
          addToast({
            type: "warning",
            title: "No se pudo cargar la imagen del candidato",
            description: err.message,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedCandidate, addToast]);

  // ============================================================
  // Canvas drawing: probe image with minutiae
  // ============================================================

  useEffect(() => {
    const canvas = probeCanvasRef.current;
    if (!canvas) return;
    const src = processedDataUrl ?? probeDataUrl;
    if (!src) return;
    const img = new Image();
    img.onload = () => {
      probeImgRef.current = img;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);

      const matchedIndices = new Set<number>();
      if (selectedCandidate) {
        for (const e of selectedCandidate.supporting_pairs) {
          matchedIndices.add(e.probe_mi_idx);
        }
      }

      // Prefer search-pipeline minutiae (same coordinate space as matchedIndices).
      // Fall back to preview minutiae when no search has been done.
      const probeMinutiae = searchResult?.probe_minutiae ?? preview?.minutiae;
      if (probeMinutiae) {
        for (let i = 0; i < probeMinutiae.length; i++) {
          const m = probeMinutiae[i];
          if (!m) continue;
          if (matchedIndices.has(i)) {
            drawDot(ctx, m.x, m.y, 7, PALETTE_HIT, PALETTE_HIT_RING, 2.5);
          } else {
            drawDot(ctx, m.x, m.y, 4, "#9ca3af", "#ffffff", 1);
          }
        }
      }
    };
    img.src = src;
  }, [processedDataUrl, probeDataUrl, preview, searchResult, selectedCandidate]);

  // ============================================================
  // Canvas drawing: candidate image with matched minutiae
  // ============================================================

  useEffect(() => {
    const canvas = candidateCanvasRef.current;
    if (!canvas) return;
    const src = candidateDataUrl ?? processedDataUrl;
    if (!src) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }
    const img = new Image();
    img.onload = () => {
      candidateImgRef.current = img;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);

      if (selectedCandidate) {
        const matched: MinutiaPoint[] = selectedCandidate.supporting_pairs.map(
          (e) => ({ x: e.candidate_mi_x * img.naturalWidth, y: e.candidate_mi_y * img.naturalHeight, angle: e.candidate_mi_angle, type: 2 }),
        );
        for (const m of matched) {
          drawDot(ctx, m.x, m.y, 7, PALETTE_HIT, PALETTE_HIT_RING, 2.5);
        }
      } else if (preview?.minutiae) {
        // Default mode: show the same processed probe with all minutiae
        // circled. Lets the perito visually confirm the extraction
        // before kicking off a search.
        for (const m of preview.minutiae) {
          drawDot(ctx, m.x, m.y, 4, "#22c55e", "#ffffff", 1.2);
        }
      }
    };
    img.src = src;
  }, [candidateDataUrl, processedDataUrl, selectedCandidate, preview]);

  // ============================================================
  // Render
  // ============================================================

  return (
    <div className="min-h-screen bg-background text-foreground p-6 font-sans dark">
      <div className="max-w-[1400px] mx-auto space-y-4">
        <header className="flex items-center justify-between border-b border-border pb-3">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate("/")}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <Fingerprint className="w-5 h-5 text-primary" />
            <h1 className="text-xl font-bold tracking-tight">Análisis de Huella</h1>
          </div>
          {latentFile && (
            <Button variant="ghost" onClick={handleReset}>
              Empezar de nuevo
            </Button>
          )}
        </header>

        {/* ============================================================ */}
        {/* WorkflowStepper — always visible. Hosts step 1's dropzone or  */}
        {/* the workbench once an image is loaded.                         */}
        {/* ============================================================ */}
        <WorkflowStepper
          current={currentStep}
          previewRunning={previewMutation.isPending}
          searchRunning={searchMutation.isPending}
          minutiae={preview?.minutiae.length ?? 0}
          candidateCount={searchResult?.candidates.length ?? 0}
          queryTimeMs={searchResult?.query_time_ms}
        >
          {!probeDataUrl ? (
            <UploadDropzone
              onFile={handleFile}
              running={previewMutation.isPending}
            />
          ) : (
            <>
            {/* Top bar: actions + status */}
            <div className="flex flex-wrap items-center gap-2 p-3 bg-card border border-border rounded-lg mb-3">
              <Button
                size="sm"
                onClick={handleSearch}
                disabled={!preview || searchMutation.isPending || preview.minutiae.length === 0}
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
                {preview ? (
                  <>{preview.minutiae.length} minucias</>
                ) : previewMutation.isPending ? (
                  <span className="text-primary">procesando…</span>
                ) : (
                  "—"
                )}
                {searchResult && (
                  <> · {searchResult.candidates.length} candidatos · {searchResult.query_time_ms}ms</>
                )}
              </div>
            </div>

            {/* Inline enroll picker */}
            {showEnrollPicker && (
              <Card className="border-border/60">
                <CardContent className="p-3 flex flex-wrap items-center gap-2">
                  <span className="text-sm text-muted-foreground">Persona:</span>
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

            {/* ============================================================ */}
            {/* THE BIG VIEW: probe image with circles (LEFT) + comparison   */}
            {/* (RIGHT). The comparison becomes the candidate when one is  */}
            {/* selected.                                                       */}
            {/* ============================================================ */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Probe image with circles */}
              <Card className="border-border/60">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
                    {searchResult ? "Procesada" : "Original"}{" "}
                    {preview && `· ${preview.minutiae.length} minucias`}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center">
                    <canvas
                      ref={probeCanvasRef}
                      className="w-full h-full object-contain"
                    />
                  </div>
                </CardContent>
              </Card>

              {/* Candidate image with matched minutiae */}
              <Card className="border-border/60">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    {selectedCandidate ? (
                      <>
                        <span>
                          Candidato · {selectedCandidate.full_name ?? selectedCandidate.external_id}
                        </span>
                        <span className="text-primary font-bold">
                          {Math.round((selectedCandidate.score ?? 0) * 100)}%
                        </span>
                      </>
                    ) : searchResult && searchResult.candidates.length === 0 ? (
                      "Sin candidatos"
                    ) : (
                      "Procesada · minucias detectadas"
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center">
                    {candidateLoading ? (
                      <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
                    ) : candidateDataUrl || processedDataUrl ? (
                      <canvas
                        ref={candidateCanvasRef}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="text-center text-muted-foreground text-sm px-4">
                        Subí una huella para empezar
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Top-10 candidate list (always shown when results arrive) */}
            {searchResult && searchResult.candidates.length > 0 && (
              <Card className="border-border/60">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                    <Trophy className="w-3.5 h-3.5" />
                    Top {searchResult.candidates.length} — click para comparar
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <ul className="divide-y divide-border/40 max-h-[480px] overflow-y-auto">
                    {searchResult.candidates.map((c, i) => {
                      const isSelected = i === selectedIdx;
                      const label =
                        c.full_name ?? c.external_id ?? c.person_id.slice(0, 8);
                      const scoreInfo = scoreColor(c.score ?? 0);
                      const barColor = scoreInfo.text.includes("green")
                        ? "#22c55e"
                        : scoreInfo.text.includes("yellow")
                        ? "#eab308"
                        : "#ef4444";
                      return (
                        <li key={c.person_id}>
                          <button
                            onClick={() => setSelectedIdx(i)}
                            className={cn(
                              "w-full text-left px-3 py-2.5 transition-all flex items-center gap-3",
                              "hover:bg-muted/40",
                              isSelected &&
                                "bg-primary/10 border-l-2 border-l-primary",
                            )}
                          >
                            <span
                              className={cn(
                                "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold",
                                isSelected
                                  ? "bg-primary text-primary-foreground"
                                  : "bg-muted text-muted-foreground",
                              )}
                            >
                              {i + 1}
                            </span>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span
                                  className={cn(
                                    "text-sm truncate",
                                    isSelected
                                      ? "font-bold"
                                      : "font-medium",
                                  )}
                                >
                                  {label}
                                </span>
                                <span
                                  className={cn(
                                    "text-[10px] uppercase tracking-wider font-bold",
                                    scoreInfo.text,
                                  )}
                                >
                                  {scoreInfo.label}
                                </span>
                                {isSelected && (
                                  <span className="ml-auto text-[10px] uppercase tracking-wider font-bold text-primary flex items-center gap-1">
                                    <Search className="w-3 h-3" />
                                    Comparando
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center gap-3 text-[11px] text-muted-foreground mt-0.5 font-mono">
                                <span
                                  className={cn("font-bold", scoreInfo.text)}
                                >
                                  {Math.round((c.score ?? 0) * 100)}%
                                </span>
                                <span>{c.peak_votes} pares</span>
                                <span>
                                  {c.supporting_pairs.length} match
                                </span>
                              </div>
                              <div className="mt-1.5 h-1 bg-muted/60 rounded overflow-hidden">
                                <div
                                  className="h-full transition-all"
                                  style={{
                                    width: `${Math.min((c.score ?? 0) * 100, 100)}%`,
                                    backgroundColor: barColor,
                                  }}
                                />
                              </div>
                            </div>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </CardContent>
              </Card>
            )}

            {/* Detail panel for selected candidate */}
            {selectedCandidate && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center gap-2 p-3 bg-card border border-border rounded-lg">
                  <Search className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">Modo comparación activo</span>
                  <span className="text-xs text-muted-foreground">
                    #{selectedIdx + 1} · {selectedCandidate.full_name ?? selectedCandidate.external_id}
                  </span>
                  <div className="ml-auto flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleOpenCreateCase(selectedCandidate)}
                    >
                      <FilePlus className="w-3.5 h-3.5 mr-1.5" />
                      Crear caso
                    </Button>
                  </div>
                </div>
                <CandidateDetailPanel
                  candidate={selectedCandidate}
                  probeImageUrl={processedDataUrl ?? probeDataUrl}
                  probeMinutiae={
                    searchResult?.probe_minutiae.map((m) => ({
                      x: m.x,
                      y: m.y,
                      angle: m.angle,
                      type: m.type,
                    })) ?? []
                  }
                  candidateImageUrl={candidateDataUrl}
                  onDismiss={() => setSelectedIdx(0)}
                />
              </div>
            )}
            </>
          )}
        </WorkflowStepper>

        {/* Create-case modal */}
        {showCreateCase && selectedCandidate && (
          <div
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
            onClick={() => !createCaseMutation.isPending && setShowCreateCase(false)}
          >
            <Card className="w-full max-w-md" onClick={(e) => e.stopPropagation()}>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <FilePlus className="w-4 h-4" />
                  Crear caso
                </CardTitle>
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
                    onClick={() => {
                      if (!selectedCandidate || !caseNumber.trim() || !caseTitle.trim()) {
                        addToast({ type: "error", title: "Campos incompletos" });
                        return;
                      }
                      createCaseMutation.mutate({
                        case_number: caseNumber.trim(),
                        title: caseTitle.trim(),
                        description: `Generado desde análisis top-level. Candidato: ${selectedCandidate.full_name ?? selectedCandidate.external_id ?? selectedCandidate.person_id}.`,
                        status: "open",
                      });
                    }}
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
      </div>
    </div>
  );
}

function drawDot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  fillColor: string,
  ringColor: string,
  ringWidth: number,
): void {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = fillColor;
  ctx.fill();
  ctx.strokeStyle = ringColor;
  ctx.lineWidth = ringWidth;
  ctx.stroke();
}

// ============================================================
// scoreColor — color classes by threshold (used by Top-10 list)
// ============================================================
function scoreColor(score: number): {
  bg: string;
  text: string;
  ring: string;
  label: string;
} {
  if (score >= MATCH_THRESHOLD_GOOD) {
    return {
      bg: "bg-green-500/15",
      text: "text-green-500",
      ring: "ring-green-500/40",
      label: "Coincidencia alta",
    };
  }
  if (score >= MATCH_THRESHOLD_FAIR) {
    return {
      bg: "bg-yellow-500/15",
      text: "text-yellow-500",
      ring: "ring-yellow-500/40",
      label: "Coincidencia media",
    };
  }
  return {
    bg: "bg-red-500/15",
    text: "text-red-500",
    ring: "ring-red-500/40",
    label: "Coincidencia baja",
  };
}

// ============================================================
// Workflow — the Stepper-as-frame component
// ============================================================
// Always visible at the top of /analisis. Hosts:
//   1. Progress bar (fills as steps complete)
//   2. 4 step indicators (Subir → Extraer → Buscar → Resultado)
//   3. Body slot — the current step's content (dropzone or workbench)

interface WorkflowStepperProps {
  current: 0 | 1 | 2 | 3;
  previewRunning: boolean;
  searchRunning: boolean;
  minutiae: number;
  candidateCount: number;
  queryTimeMs: number | undefined;
  children: React.ReactNode;
}

function WorkflowStepper({
  current,
  previewRunning,
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
      status: previewRunning
        ? "Procesando Gabor + minucias…"
        : minutiae > 0
        ? `${minutiae} minucias detectadas`
        : current < 1
        ? "Esperando imagen"
        : "Listo para procesar",
      running: previewRunning,
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
        current >= 3
          ? "Comparación activa"
          : current === 2
          ? "Selecciona un candidato"
          : "Pendiente",
      running: false,
    },
  ];

  const progressPct = (current / 4) * 100;

  return (
    <Card className="overflow-hidden border-border/60">
      {/* Progress bar */}
      <div className="h-1.5 bg-muted overflow-hidden">
        <div
          className={cn(
            "h-full transition-all duration-700 ease-out",
            previewRunning || searchRunning
              ? "bg-primary/60 animate-pulse"
              : "bg-primary",
          )}
          style={{ width: `${progressPct}%` }}
        />
      </div>

      {/* Step indicators */}
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

      {/* Body slot */}
      <div className="p-4 bg-background/30">{children}</div>
    </Card>
  );
}

// ============================================================
// UploadDropzone — step 1's body
// ============================================================

interface UploadDropzoneProps {
  onFile: (file: File) => void;
  running?: boolean;
}

function UploadDropzone({
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
