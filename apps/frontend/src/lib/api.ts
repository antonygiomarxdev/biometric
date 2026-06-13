/**
 * API client for the v1 forensic workflow endpoints.
 *
 * Uses fetch directly (the generated OpenAPI client does not cover
 * the v1 routers added in Phase 1 refactoring).
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
// Domain types (mirrors backend Pydantic models)
// ---------------------------------------------------------------------------

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

export interface MatchCandidate {
  person_id: string;
  name: string;
  document: string;
  evidence_id: string;
  l2_distance: number;
  score: number;
}

export interface MatchSearchResponse {
  success: boolean;
  top_k: number;
  candidates: MatchCandidate[];
  total: number;
}

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

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

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

export function listEvidence(
  caseId?: string,
  skip = 0,
  limit = 20,
): Promise<EvidenceListResponse> {
  return request<EvidenceListResponse>("GET", "/api/v1/evidence", {
    query: { case_id: caseId, skip, limit },
  });
}

export function searchMatching(
  file: File,
  topK = 10,
): Promise<MatchSearchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<MatchSearchResponse>("POST", "/api/v1/matching/search", {
    formData,
    query: { top_k: topK },
  });
}

export function createDecision(
  decision: DecisionCreate,
): Promise<DecisionResponse> {
  return request<DecisionResponse>("POST", "/api/v1/decisions", {
    body: decision,
  });
}
