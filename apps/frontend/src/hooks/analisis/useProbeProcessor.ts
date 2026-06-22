import { useState, useCallback, useEffect } from "react";
import { useToast } from "@/components/ui/toast";

const VALID_TYPES = ["image/bmp", "image/png", "image/jpeg", "image/jpg", "image/tiff", "image/tif", "image/x-tiff"];
const MAX_BYTES = 10 * 1024 * 1024;

/**
 * Phase 29: the probe no longer goes through a minutia-extraction
 * step.  This hook just decodes the file into a local data URL for
 * preview and exposes a ``latentFile`` for the search call.
 */
export function useProbeProcessor() {
  const { addToast } = useToast();
  const [latentFile, setLatentFile] = useState<File | null>(null);
  const [probeDataUrl, setProbeDataUrl] = useState<string | null>(null);

  useEffect(() => {
    let url = probeDataUrl;
    return () => {
      if (url && url.startsWith("blob:")) {
        URL.revokeObjectURL(url);
      }
    };
  }, [probeDataUrl]);

  const handleFile = useCallback(
    (file: File) => {
      if (!VALID_TYPES.includes(file.type)) {
        addToast({
          type: "error",
          title: "Tipo inválido",
          description: "BMP, PNG, JPEG o TIFF",
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

      const localUrl = URL.createObjectURL(file);
      setProbeDataUrl(localUrl);
      setLatentFile(file);
    },
    [addToast]
  );

  const reset = useCallback(() => {
    setLatentFile(null);
    setProbeDataUrl(null);
  }, []);

  return {
    latentFile,
    probeDataUrl,
    handleFile,
    reset,
  };
}
