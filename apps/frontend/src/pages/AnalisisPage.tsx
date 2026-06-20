import {
  WorkflowStepper,
  UploadDropzone,
} from "@/components/analisis/WorkflowStepper";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { CandidateDetailPanel } from "@/components/fingerprint/CandidateDetailPanel";
import { drawMinutiaMarker } from "@/hooks/useMatchCanvas";
import { useToast } from "@/components/ui/toast";
import {
  searchMatching,
  listPersons,
  createFingerprintSlot,
  enrollFingerprint,
  createCase,
  getMinutiaeForImage,
  type MatchCandidate,
  type MatchSearchResponse,
  type MinutiaPoint,
  type CaseCreateInput,
} from "@/lib/api";
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
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState, useCallback } from "react";

const VALID_TYPES = ["image/bmp", "image/png", "image/jpeg", "image/jpg"];
const MAX_BYTES = 10 * 1024 * 1024;

const PALETTE_HIT = "#ffffff";

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
  const [probePreviewUrl, setProbePreviewUrl] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<MatchSearchResponse | null>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);

  const [showEnrollPicker, setShowEnrollPicker] = useState(false);
  const [enrollPersonId, setEnrollPersonId] = useState("");
  const [enrolling, setEnrolling] = useState(false);
  const [showCreateCase, setShowCreateCase] = useState(false);
  const [caseNumber, setCaseNumber] = useState("");
  const [caseTitle, setCaseTitle] = useState("");

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
          title: `${result.candidates.length} candidato${
            result.candidates.length !== 1 ? "s" : ""
          } encontrado${result.candidates.length !== 1 ? "s" : ""}`,
          description: "Hacé click en uno para ver la comparación",
          duration: 4000,
        });
      }
    },
    onError: (err: Error) => {
      addToast({
        type: "error",
        title: "Error en búsqueda",
        description: err.message,
      });
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

  const handleFile = useCallback(
    (file: File) => {
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
          title: "Archivo grande",
          description: "Máx 10MB",
        });
        return;
      }
      const reader = new FileReader();
      reader.onload = (ev) => {
        setProbeDataUrl(ev.target?.result as string);
        setLatentFile(file);
        setSearchResult(null);
        setSelectedIdx(0);
        setProbePreviewUrl(null);
      };
      reader.readAsDataURL(file);

      getMinutiaeForImage(file)
        .then((res) => {
          setProbePreviewUrl(res.processed_image_url);
        })
        .catch((err) => {
          console.error("Preview failed:", err);
        });
    },
    [addToast]
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
    setLatentFile(null);
    setProbeDataUrl(null);
    setSearchResult(null);
    setSelectedIdx(0);
  }, []);

  const selectedCandidate: MatchCandidate | null =
    searchResult?.candidates[selectedIdx] ?? null;

  const currentStep: 0 | 1 | 2 = (() => {
    if (searchResult) return 2;
    if (probeDataUrl) return 1;
    return 0;
  })();

  const probeSrc = searchResult?.probe_image_url ?? probePreviewUrl ?? probeDataUrl;

  useEffect(() => {
    const canvas = probeCanvasRef.current;
    if (!canvas) return;
    if (!probeSrc) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
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

      const probeMinutiae = searchResult?.probe_minutiae;
      if (probeMinutiae) {
        for (let i = 0; i < probeMinutiae.length; i++) {
          const m = probeMinutiae[i];
          if (!m) continue;
          const color = matchedIndices.has(i)
            ? PALETTE_HIT
            : "rgba(255,255,255,0.7)";
          drawMinutiaMarker(ctx, m, color);
        }
      }
    };
    img.src = probeSrc;
  }, [probeSrc, searchResult, selectedCandidate]);

  const candidateSrc = selectedCandidate?.image_url ?? probePreviewUrl ?? null;

  useEffect(() => {
    const canvas = candidateCanvasRef.current;
    if (!canvas) return;
    if (!candidateSrc) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
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
        for (const e of selectedCandidate.supporting_pairs) {
          drawMinutiaMarker(
            ctx,
            {
              x: e.candidate_mi_x,
              y: e.candidate_mi_y,
              angle: e.candidate_mi_angle,
              type: 2,
            },
            PALETTE_HIT
          );
        }
      }
    };
    img.src = candidateSrc;
  }, [candidateSrc, selectedCandidate]);

  // ... rest of the render function
}
