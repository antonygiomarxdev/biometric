import type { IdentifyResponse, ExtractResponse } from "../client";

export type BiometricModality = "fingerprint" | "face" | "iris";

export interface FingerprintItem {
  id: string;
  file: File;
  preview: string;
  status: "pending" | "processing" | "completed" | "error";
  result?: IdentifyResponse;
  extractData?: ExtractResponse;
  referenceImageUrl?: string;
}

export type AppMode = "scan" | "register";
