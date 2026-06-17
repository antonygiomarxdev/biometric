import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Upload,
  Fingerprint,
  Loader2,
  XCircle,
  CheckCircle2,
  ArrowLeft,
  UserPlus,
  FilePlus,
  Search,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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

export default function AnalisisPage() {
  const navigate = useNavigate();
  const { addToast } = useToast();

  const probeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const candidateCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const probeImgRef = useRef<HTMLImageElement | null>(null);
  const candidateImgRef = useRef<HTMLImageElement | null>(null);

  const [latentFile, setLatentFile] = useState<File | null>(null);
  const [probeDataUrl, setProbeDataUrl] = useState<string | null>(null);
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

  useEffect(() => {
    if (!selectedCandidate) {
      setCandidateDataUrl(null);
      return;
    }
    const captureId = selectedCandidate.match_trace[0]?.candidate_capture_id;
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
      .catch((err) => {
        if (!cancelled) {
          setCandidateLoading(false);
          // 503 is expected for legacy captures; degrade gracefully
          if (!String(err.message).includes("503")) {
            addToast({
              type: "warning",
              title: "No se pudo cargar la imagen del candidato",
              description: "Re-enrolá la captura para tener la imagen enhanced.",
            });
          }
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
    if (!canvas || !probeDataUrl) return;
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
        for (const e of selectedCandidate.match_trace) {
          matchedIndices.add(e.probe_cylinder_index);
        }
      }

      if (preview?.minutiae) {
        for (let i = 0; i < preview.minutiae.length; i++) {
          const m = preview.minutiae[i];
          if (!m) continue;
          if (matchedIndices.has(i)) {
            drawDot(ctx, m.x, m.y, 7, PALETTE_HIT, PALETTE_HIT_RING, 2.5);
          } else {
            drawDot(ctx, m.x, m.y, 4, "#9ca3af", "#ffffff", 1);
          }
        }
      }
    };
    img.src = probeDataUrl;
  }, [probeDataUrl, preview, selectedCandidate]);

  // ============================================================
  // Canvas drawing: candidate image with matched minutiae
  // ============================================================

  useEffect(() => {
    const canvas = candidateCanvasRef.current;
    if (!canvas) return;
    if (!candidateDataUrl) {
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
        const matched: MinutiaPoint[] = selectedCandidate.match_trace.map(
          (e) => ({ x: e.candidate_x, y: e.candidate_y, angle: e.candidate_angle, type: 2 }),
        );
        for (const m of matched) {
          drawDot(ctx, m.x, m.y, 7, PALETTE_HIT, PALETTE_HIT_RING, 2.5);
        }
      }
    };
    img.src = candidateDataUrl;
  }, [candidateDataUrl, selectedCandidate]);

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
        {/* If no image yet, show the upload zone                          */}
        {/* ============================================================ */}
        {!probeDataUrl && (
          <Card
            className="border-2 border-dashed border-border hover:border-primary/50 transition-colors cursor-pointer"
            onClick={() => document.getElementById("file-input")?.click()}
          >
            <CardContent className="flex flex-col items-center justify-center py-16">
              <Upload className="w-12 h-12 mb-3 text-muted-foreground opacity-50" />
              <h2 className="text-lg font-semibold mb-1">Subí la huella latente</h2>
              <p className="text-sm text-muted-foreground">
                BMP, PNG o JPEG. Máx 10MB.
              </p>
              <input
                id="file-input"
                type="file"
                className="hidden"
                accept="image/*"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />
            </CardContent>
          </Card>
        )}

        {/* ============================================================ */}
        {/* The image is loaded: show the workbench                          */}
        {/* ============================================================ */}
        {probeDataUrl && (
          <>
            {/* Top bar: actions + status */}
            <div className="flex flex-wrap items-center gap-2 p-3 bg-card border border-border rounded-lg">
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
                    Latente {preview && `· ${preview.minutiae.length} minucias`}
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
                          {Math.round(selectedCandidate.total_score * 100)}%
                        </span>
                      </>
                    ) : searchResult && searchResult.candidates.length === 0 ? (
                      "Sin candidatos"
                    ) : (
                      "Candidato (selecciona uno)"
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="aspect-square bg-black rounded overflow-hidden flex items-center justify-center">
                    {candidateLoading ? (
                      <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
                    ) : candidateDataUrl ? (
                      <canvas
                        ref={candidateCanvasRef}
                        className="w-full h-full object-contain"
                      />
                    ) : searchResult ? (
                      <div className="text-center text-muted-foreground text-sm px-4">
                        {searchResult.candidates.length === 0
                          ? "Esta huella no tiene match. Probá enrolar."
                          : "Click en un candidato para ver la comparación"}
                      </div>
                    ) : (
                      <div className="text-center text-muted-foreground text-sm px-4">
                        Buscá para ver candidatos
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Candidates strip */}
            {searchResult && searchResult.candidates.length > 0 && (
              <Card className="border-border/60">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground">
                    Candidatos ({searchResult.candidates.length})
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                  <div className="flex flex-wrap gap-2">
                    {searchResult.candidates.map((c, i) => {
                      const isSelected = i === selectedIdx;
                      const label = c.full_name ?? c.external_id ?? c.person_id.slice(0, 8);
                      return (
                        <button
                          key={c.person_id}
                          onClick={() => setSelectedIdx(i)}
                          className={`
                            px-3 py-2 rounded border text-left transition-all
                            ${isSelected
                              ? "border-primary bg-primary/10 ring-1 ring-primary"
                              : "border-border hover:border-primary/50 bg-card/50"
                            }
                          `}
                        >
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground font-mono">#{i + 1}</span>
                            <span className="text-sm font-medium">{label}</span>
                            <span
                              className="text-sm font-bold"
                              style={{
                                color:
                                  c.total_score >= 0.8
                                    ? "#22c55e"
                                    : c.total_score >= 0.5
                                    ? "#eab308"
                                    : "#9ca3af",
                              }}
                            >
                              {Math.round(c.total_score * 100)}%
                            </span>
                          </div>
                          <div className="text-[10px] text-muted-foreground">
                            {c.hits} cilindros matched
                          </div>
                        </button>
                      );
                    })}
                  </div>

                  {selectedCandidate && (
                    <div className="mt-3 flex items-center gap-2 pt-3 border-t border-border/40">
                      <span className="text-xs text-muted-foreground">
                        Match trace: <span className="font-mono text-foreground">{selectedCandidate.match_trace.length}</span> pares
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
                  )}
                </CardContent>
              </Card>
            )}
          </>
        )}

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
