import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { useToast } from "@/components/ui/toast";
import { searchMatching, type MatchSearchResponse } from "@/lib/api";

export function useSearchManager() {
  const { addToast } = useToast();
  const [searchResult, setSearchResult] = useState<MatchSearchResponse | null>(
    null
  );

  const searchMutation = useMutation({
    mutationFn: (file: File) => searchMatching(file, 10),
    onSuccess: (result) => {
      setSearchResult(result);
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

  const reset = useCallback(() => {
    setSearchResult(null);
  }, []);

  return {
    searchResult,
    searchMutation,
    reset,
  };
}
