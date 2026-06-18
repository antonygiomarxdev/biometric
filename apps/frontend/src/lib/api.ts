/**
 * API client for the v1 forensic workflow endpoints (Phase 23 rewrite).
 *
 * Single source of truth for backend communication (D-28). The types
 * mirror the Pydantic v1 models in apps/backend/src/schemas/ and
 * apps/backend/src/core/types.py; field names are snake_case to match
 * the wire format exactly.
 *
 * Per D-14, this file is hand-maintained. The previous generated
 * client at @/client is being deleted in Plan 23-07.
 */

// ---------------------------------------------------------------------------
// Base configuration
// ---------------------------------------------------------------------------

const API_BASE = "http://localhost:8000";

async function request<T>(
  method: string,
  path: string,
  options?: {
    body?: unknown;
    query?: Record<string, string | number | undefined>;
    formData?: FormData;
  },
): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (options?.query) {
    for (const [key, value] of Object.entries(options.query)) {
      if (value !== undefined) {
        url.searchParams.set(key, String(value));
      }
    }
  }

  const headers: Record<string, string> = {};
  let body: BodyInit | undefined;

  if (options?.formData) {
    // multipart/form-data — let the browser set Content-Type with boundary
    body = options.formData;
  } else if (options?.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.body);
  }

  const response = await fetch(url.toString(), { method, headers, body });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new ApiError(response.status, errorBody);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Error class
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number;
  body: string;

  constructor(status: number, body: string) {
    super(`API error ${status}: ${body}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

// ---------------------------------------------------------------------------
// Domain types — mirror backend Pydantic models
// ---------------------------------------------------------------------------

/**
 * A single minutia point. Mirrors the legacy MinutiaPoint from the
 * deleted src/client/ and the FingerprintPreviewResponse from the
 * backend. Used by useCanvasDrawer and MinutiaeEditor.
 */
export interface MinutiaPoint {
  x: number;
  y: number;
  angle: number; // radians
  type: number; // 0=termination, 1=bifurcation, 2=unknown
}

/** A single supporting pair from pair-based matching (Phase 24). */
export interface SupportingPair {
  probe_pair_index: number;
  probe_mi_idx: number;
  probe_mj_idx: number;
  probe_mi_x: number;
  probe_mi_y: number;
  candidate_mi_x: number;
  candidate_mi_y: number;
  candidate_mi_angle: number;
  candidate_mj_x: number;
  candidate_mj_y: number;
  candidate_mj_angle: number;
  similarity: number;
}

/** A single ranked match candidate from pair-based matching (Phase 24). */
export interface MatchCandidate {
  person_id: string;
  score: number;
  peak_votes: number;
  peak_transformation: {
    dx: number;
    dy: number;
    dtheta: number;
  };
  supporting_pairs: SupportingPair[];
  num_probe_pairs: number;
  full_name: string | null;
  external_id: string | null;
}

/** Response of POST /api/v1/matching/search (Phase 24). */
export interface MatchSearchResponse {
  success: boolean;
  query_time_ms: number;
  total_candidates: number;
  probe_minutiae: MinutiaPoint[];
  candidates: MatchCandidate[];
}

/** Person entity (Phase 17). */
export interface PersonResponse {
  id: string;
  external_id: string | null;
  full_name: string | null;
  doc_type: string | null;
  doc_number: string | null;
  sex: "M" | "F" | "X" | null;
  dob: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

/** Fingerprint slot (Phase 17). */
export interface FingerprintSlotResponse {
  id: string;
  person_id: string;
  finger_position: number;
  capture_type: string;
  capture_count: number;
  first_captured_at: string | null;
  last_captured_at: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

/** Single capture record (Phase 17). */
export interface CaptureResponse {
  id: string;
  fingerprint_id: string;
  capture_index: number;
  image_uri: string;
  image_dpi: number | null;
  image_quality_score: number | null;
  algorithm_version: string;
  processed_at: string;
  num_minutiae: number | null;
  num_graphs: number | null;
  is_reference: boolean;
  is_exemplar: boolean;
  notes: string | null;
  graphs: unknown[]; // RidgeGraph removed in Phase 23
}

/** Response of POST /api/v1/fingerprints/preview (Phase 23). */
export interface FingerprintPreviewResponse {
  processed_image: string; // base64 PNG (no data: prefix)
  minutiae: MinutiaPoint[];
  terminations: number;
  bifurcations: number;
  image_shape: [number, number]; // [h, w]
  image_dtype: string;
}

/** POST /api/v1/decisions body. */
export interface DecisionCreate {
  case_id: string;
  evidence_id: string | null;
  verdict: string;
  comments: string | null;
}

export interface DecisionResponse {
  id: string;
  case_id: string;
  evidence_id: string | null;
  verdict: string;
  comments: string | null;
  created_at: string;
}

/** Forensic case (Phase 18). */
export interface CaseResponse {
  id: string;
  case_number: string;
  title: string;
  description: string | null;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface CaseListResponse {
  items: CaseResponse[];
  total: number;
  skip: number;
  limit: number;
}

/** Latent fingerprint evidence attached to a case (Phase 18). */
export interface EvidenceResponse {
  id: string;
  case_id: string;
  fingerprint_id: string;
  image_path: string | null;
  num_minutiae: number | null;
  created_at: string;
  updated_at: string;
}

export interface EvidenceListResponse {
  items: EvidenceResponse[];
  total: number;
  skip: number;
  limit: number;
}

// ---------------------------------------------------------------------------
// API functions — single point of contact with backend
// ---------------------------------------------------------------------------

// Cases
export function listCases(
  status?: string,
  skip = 0,
  limit = 20,
): Promise<CaseListResponse> {
  return request<CaseListResponse>("GET", "/api/v1/cases", {
    query: { status, skip, limit },
  });
}

export function getCase(caseId: string): Promise<CaseResponse> {
  return request<CaseResponse>("GET", `/api/v1/cases/${caseId}`);
}

// Evidence
export function listEvidence(
  caseId?: string,
  skip = 0,
  limit = 20,
): Promise<EvidenceListResponse> {
  return request<EvidenceListResponse>("GET", "/api/v1/evidence", {
    query: { case_id: caseId, skip, limit },
  });
}

// Persons (Phase 23, D-29)
export function listPersons(
  skip = 0,
  limit = 100,
  search?: string,
): Promise<PersonResponse[]> {
  return request<PersonResponse[]>("GET", "/api/v1/persons", {
    query: { skip, limit, search },
  });
}

export function getPerson(id: string): Promise<PersonResponse> {
  return request<PersonResponse>("GET", `/api/v1/persons/${id}`);
}

// Fingerprint slots
export function createFingerprintSlot(
  personId: string,
  fingerPosition: number,
  captureType = "rolled",
  notes?: string,
): Promise<FingerprintSlotResponse> {
  return request<FingerprintSlotResponse>(
    "POST",
    `/api/v1/persons/${personId}/fingerprints`,
    {
      body: {
        finger_position: fingerPosition,
        capture_type: captureType,
        notes: notes ?? null,
      },
    },
  );
}

export function listFingerprintsForPerson(
  personId: string,
): Promise<FingerprintSlotResponse[]> {
  return request<FingerprintSlotResponse[]>(
    "GET",
    `/api/v1/persons/${personId}/fingerprints`,
  );
}

// Pre-enrollment preview (Phase 23, D-29)
export function getMinutiaeForImage(
  file: File,
): Promise<FingerprintPreviewResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<FingerprintPreviewResponse>(
    "POST",
    "/api/v1/fingerprints/preview",
    { formData },
  );
}

// Enrollment (POST capture)
export function enrollFingerprint(
  fingerprintId: string,
  file: File,
  options?: {
    imageDpi?: number;
    isReference?: boolean;
    isExemplar?: boolean;
    notes?: string;
  },
): Promise<CaptureResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (options?.imageDpi !== undefined) {
    formData.append("image_dpi", String(options.imageDpi));
  }
  if (options?.isReference !== undefined) {
    formData.append("is_reference", String(options.isReference));
  }
  formData.append("is_exemplar", String(options?.isExemplar ?? true));
  if (options?.notes) {
    formData.append("notes", options.notes);
  }
  return request<CaptureResponse>(
    "POST",
    `/api/v1/fingerprints/${fingerprintId}/captures`,
    { formData },
  );
}

// Matching (Phase 23, returns MatchSearchResponse with match_trace + probe_minutiae)
export function searchMatching(
  file: File,
  topK = 10,
): Promise<MatchSearchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<MatchSearchResponse>(
    "POST",
    "/api/v1/matching/search",
    { formData, query: { top_k: topK } },
  );
}

/** Fetch the Gabor-enhanced PNG bytes for an enrolled capture (Phase 23).
 *  Returns a blob URL ready to bind to <img src=...>. */
export async function fetchCaptureImage(captureId: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/v1/captures/${captureId}/image`);
  if (!res.ok) {
    const err = new Error(
      `Failed to fetch capture image: ${res.status}`,
    ) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// Decisions
export function createDecision(
  decision: DecisionCreate,
): Promise<DecisionResponse> {
  return request<DecisionResponse>("POST", "/api/v1/decisions", {
    body: decision,
  });
}

// Cases (used by /search "Crear caso desde match" flow)
export interface CaseCreateInput {
  case_number: string;
  title: string;
  description?: string | null;
  status?: "open" | "closed" | "archived";
}

export function createCase(
  body: CaseCreateInput,
): Promise<CaseResponse> {
  return request<CaseResponse>("POST", "/api/v1/cases", { body });
}
