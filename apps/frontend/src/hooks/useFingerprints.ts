import { useState, useCallback } from "react";
import { DefaultService, OpenAPI } from "../client";
import type {
  FingerprintItem,
  AppMode,
  BiometricModality,
} from "../types/fingerprint";
import { useToast } from "@/components/ui/toast";

// Ensure API Base URL is set
OpenAPI.BASE = "http://localhost:8000";

export const useFingerprints = () => {
  const [fingerprints, setFingerprints] = useState<FingerprintItem[]>([]);
  const [selectedFingerId, setSelectedFingerId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeMode, setActiveMode] = useState<AppMode>("scan");
  const [activeModality, setActiveModality] =
    useState<BiometricModality>("fingerprint");

  // Registration State
  const [regName, setRegName] = useState("");
  const [regId, setRegId] = useState("");

  const { addToast } = useToast();

  const addFingerprints = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;

      Array.from(files).forEach((file) => {
        // Validate type and size
        if (
          !["image/bmp", "image/png", "image/jpeg", "image/jpg"].includes(
            file.type
          )
        ) {
          return;
        }
        if (file.size > 10 * 1024 * 1024) return;

        const reader = new FileReader();
        reader.onload = (e) => {
          const id = Math.random().toString(36).substring(7);
          const item: FingerprintItem = {
            id,
            file,
            preview: e.target?.result as string,
            status: "pending",
          };

          setFingerprints((prev) => {
            const updated = [...prev, item];
            // Select the first one if none selected
            if (!selectedFingerId && prev.length === 0) setSelectedFingerId(id);
            return updated;
          });

          // If it's the first item being added, select it
          if (fingerprints.length === 0) {
            setSelectedFingerId(id);
          }
        };
        reader.readAsDataURL(file);
      });
    },
    [fingerprints.length, selectedFingerId]
  );

  const processFingerprint = async (item: FingerprintItem) => {
    try {
      // Update status to processing
      setFingerprints((prev) =>
        prev.map((f) => (f.id === item.id ? { ...f, status: "processing" } : f))
      );

      // 1. Extract minutiae
      const extractRes = await DefaultService.extractMinutiaeExtractPost({
        file: item.file,
      });

      // Update with extraction data
      setFingerprints((prev) =>
        prev.map((f) =>
          f.id === item.id ? { ...f, extractData: extractRes } : f
        )
      );

      // Mostrar mensaje de éxito de procesamiento
      addToast({
        type: "info",
        title: "Procesamiento completado",
        description: `minutiae: ${extractRes.minutiae_count}, terminaciones: ${extractRes.terminations}, bifurcaciones: ${extractRes.bifurcations}`,
        duration: 3000,
      });

      if (activeMode === "scan") {
        const res = await DefaultService.identifyFingerprintIdentifyPost({
          file: item.file,
        });

        // Build reference image URL if matched
        let refImageUrl = undefined;
        if (res.matched && res.person_id) {
          refImageUrl = `${OpenAPI.BASE}/fingerprints/${res.person_id}/image`;
        }

        setFingerprints((prev) =>
          prev.map((f) =>
            f.id === item.id
              ? {
                  ...f,
                  status: "completed",
                  result: res,
                  referenceImageUrl: refImageUrl,
                }
              : f
          )
        );

        if (res.matched) {
          addToast({
            type: "success",
            title: "Identidad confirmada",
            description: `${res.name || "Desconocido"} - Confianza: ${(
              (res.score || 0) * 100
            ).toFixed(1)}%`,
          });
        }
      } else {
        // Registration Mode
        if (!regId || !regName)
          throw new Error("Datos de registro incompletos");

        await DefaultService.registerFingerprintRegisterPost({
          person_id: regId,
          name: regName,
          document: regId,
          file: item.file,
        });

        setFingerprints((prev) =>
          prev.map((f) =>
            f.id === item.id ? { ...f, status: "completed" } : f
          )
        );
        addToast({
          type: "success",
          title: "Registrado",
          description: `Huella de ${regName} guardada correctamente`,
        });
      }
    } catch (error) {
      console.error(error);
      setFingerprints((prev) =>
        prev.map((f) => (f.id === item.id ? { ...f, status: "error" } : f))
      );
      addToast({
        type: "error",
        title: "Error",
        description: "Falló el procesamiento de la huella",
      });
    }
  };

  const processAll = async () => {
    setLoading(true);
    const pending = fingerprints.filter((f) => f.status === "pending");

    // Process sequentially
    for (const item of pending) {
      await processFingerprint(item);
    }
    setLoading(false);
  };

  const selectedFingerprint = fingerprints.find(
    (f) => f.id === selectedFingerId
  );

  return {
    fingerprints,
    selectedFingerId,
    setSelectedFingerId,
    selectedFingerprint,
    addFingerprints,
    processAll,
    loading,
    activeMode,
    setActiveMode,
    activeModality,
    setActiveModality,
    registration: {
      name: regName,
      setName: setRegName,
      id: regId,
      setId: setRegId,
    },
  };
};
