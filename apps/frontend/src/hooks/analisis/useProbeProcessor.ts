import { useState, useCallback } from "react";
import { useToast } from "@/components/ui/toast";
import { getMinutiaeForImage, type MinutiaPoint } from "@/lib/api";

const VALID_TYPES = ["image/bmp", "image/png", "image/jpeg", "image/jpg"];
const MAX_BYTES = 10 * 1024 * 1024;

export function useProbeProcessor() {
  const { addToast } = useToast();
  const [latentFile, setLatentFile] = useState<File | null>(null);
  const [probeDataUrl, setProbeDataUrl] = useState<string | null>(null);
  const [probePreviewUrl, setProbePreviewUrl] = useState<string | null>(null);
  const [probeMinutiae, setProbeMinutiae] = useState<MinutiaPoint[]>([]);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (!VALID_TYPES.includes(file.type)) {
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
        setProbePreviewUrl(null);
        setProbeMinutiae([]);
      };
      reader.readAsDataURL(file);

      setIsPreviewLoading(true);
      getMinutiaeForImage(file)
        .then((res) => {
          setProbePreviewUrl(res.processed_image_url);
          setProbeMinutiae(res.minutiae);
        })
        .catch((err) => {
          console.error("Preview failed:", err);
          addToast({
            type: "error",
            title: "Error en la previsualización",
            description: err.message,
          });
        })
        .finally(() => {
          setIsPreviewLoading(false);
        });
    },
    [addToast]
  );

  const reset = useCallback(() => {
    setLatentFile(null);
    setProbeDataUrl(null);
    setProbePreviewUrl(null);
    setProbeMinutiae([]);
  }, []);

  return {
    latentFile,
    probeDataUrl,
    probePreviewUrl,
    probeMinutiae,
    isPreviewLoading,
    handleFile,
    reset,
  };
}
