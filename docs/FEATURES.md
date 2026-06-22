# System Features Map — Biometric v2.1

> **Last updated:** 2026-06-22
>
> One source of truth for what exists, what's partial, and what's missing.
> Use this to prioritise the next phases. Categories are independent;
> each area can be shipped separately.

**Legend:** ✅ Built | 🟡 Partial | ❌ Missing | 📋 Planned (SPEC exists)

---

## 1. Personas y Autenticación

| Feature | Estado | Notas |
|---------|--------|-------|
| Login (JWT) | ✅ | `POST /auth/login`, `GET /auth/me` |
| Roles (perito, admin, auditor) | 🟡 | Modelo tiene roles pero no hay RBAC middleware. Cualquier usuario puede hacer cualquier cosa. |
| Register user | ❌ | No hay endpoint de registro. Usuarios se crean via migration/seed. |
| Profile | ❌ | No hay `PUT /auth/me`. Nombre, email, configuración. |
| Per-user views | ❌ | Cada usuario ve exactamente lo mismo (Dashboard). No hay vistas separadas por rol. |

**Dependencias:** Auth router exists, but thin.

---

## 2. Personas (registro civil / base de datos)

| Feature | Estado | Notas |
|---------|--------|-------|
| List persons | ✅ | `GET /persons` con paginación |
| Get person by ID | ✅ | `GET /persons/{id}` |
| Create person | ✅ | `POST /persons` (idempotent) |
| Lookup by external_id | ✅ | `GET /persons/by-external-id/{id}` |
| Update person | ❌ | `PUT /persons/{id}` no existe. |
| Delete person | ❌ | No hay delete (forense: no se borran registros). |
| Person photo | ❌ | No hay asociación de foto de perfil. |
| Person metadata (edad, sexo, etc.) | ❌ | Solo `full_name` + `external_id`. |

---

## 3. Enrollment (alta de huellas dactilares)

| Feature | Estado | Notas |
|---------|--------|-------|
| Fingerprint slots | ✅ | `POST /fingerprints` crea slot (person_id + position) |
| Upload capture image | ✅ | `POST /fingerprints/{id}/captures` → MinIO + embedding |
| GradCAM on enrollment | ❌ | Solo en search. Enrolamiento no computa GradCAM (ahorra backward pass). |
| Interactive UI enrollment | ✅ | 3-step wizard en EnrollPage.tsx |
| Replay safety | ✅ | Idempotent por `image_hash_sha256` |
| Bulk enrollment | ✅ | `quick_enroll.py` con asyncio.Semaphore(16) |
| Batch re-enrollment | ❌ | No hay script de re-enrolamiento masivo |

---

## 4. Matching / Búsqueda

| Feature | Estado | Notas |
|---------|--------|-------|
| Search 1:N | ✅ | `POST /matching/search` → Afrika-Net embedding + Qdrant KNN |
| GradCAM | ✅ | Backward hook por búsqueda |
| U-Net enhance | 🟡 | Model cargado pero no wired al endpoint (`?enhance` es TODO) |
| Multi-finger match UI | ✅ | **Acabamos de implementar** (Plan 30-A, grouped by person) |
| Candidate comparison | ✅ | CandidateDetailPanel con side-by-side images |
| Anotación manual | ❌ | **Plan 30-B deferred.** Perito no puede marcar puntos. |
| Filter candidates | 🟡 | Server-side confidence threshold. No hay filtro por mano/dedo. |
| Search history | ❌ | No hay log de búsquedas anteriores para el perito. |
| Batch search | ❌ | Sólo 1 probe a la vez. |

---

## 5. Casos (forensic case management)

| Feature | Estado | Notas |
|---------|--------|-------|
| List cases | ✅ | `GET /cases` |
| Get case | ✅ | `GET /cases/{id}` |
| Create case | ✅ | `POST /cases` |
| Update case | ✅ | `PUT /cases/{id}` |
| Delete case | ✅ | `DELETE /cases/{id}` |
| Case detail page | 🟡 | Existe `ComparisonView.tsx` para ver caso. No hay "Case dashboard" con resumen. |
| Attach evidence to case | 🟡 | `POST /cases/{id}/evidence` |
| Annotations on case | ❌ | El perito no puede anotar sobre las imágenes. |
| Notes / comments | ❌ | No hay campo de notas del perito por caso. |
| Case status (open/closed) | ❌ | No hay estado de workflow (pendiente, revisión, cerrado). |

---

## 6. Evidence (evidencia del caso)

| Feature | Estado | Notas |
|---------|--------|-------|
| List evidence | ✅ | `GET /evidence` |
| Create evidence | ✅ | El probe se asocia como evidencia. |
| Image retrieval | ✅ | `GET /captures/{id}/image` (MinIO) |
| Multiple evidence per case | 🟡 | Modelo soporta, pero UI no permite añadir más de 1 probe. |
| Evidence chain of custody | 🟡 | Audit log existe pero no visible en UI del perito. |

---

## 7. Dictamen / Reporte PDF

| Feature | Estado | Notas |
|---------|--------|-------|
| Generate report | ✅ | `POST /reports` (endpoint existe) |
| PDF download | 🟡 | Endpoint existe pero no hay data de anotaciones perito. |
| Template | ❌ | Sin diseño de template forense. |
| Reporte con annotations | ❌ | Sin annotations (Plan 30-B). |
| Report history | ❌ | No hay listado de dictámenes generados. |

---

## 8. Decisiones (perito decide match/no-match)

| Feature | Estado | Notas |
|---------|--------|-------|
| Record decision | ✅ | `POST /decisions` |
| Decision types | ✅ | `match`, `non_match`, `inconclusive` |
| Decision evidence | ❌ | No se asocian annotations con la decisión. Solo metadata. |
| Decision audit | ✅ | Audit log para cada decisión. |

---

## 9. Auditoría y Cadena de Custodia

| Feature | Estado | Notas |
|---------|--------|-------|
| Audit log | ✅ | `GET /audit/logs` |
| Inmutable hash chain | ✅ | SELECT FOR UPDATE + hash chain. |
| Audit visible in UI | ❌ | No hay página de auditoría en frontend. |
| User activity log | 🟡 | Capturado pero no hay dashboard de actividad. |

---

## 10. IA Generativa (LLM)

| Feature | Estado | Notas |
|---------|--------|-------|
| LLM assistant | ✅ | `POST /genai/ask` |
| Report draft | ✅ | `POST /genai/report-draft` |
| LLM model | ✅ | Local (Ollama/Llama) configurable. |

---

## 11. Administración y Configuración

| Feature | Estado | Notas |
|---------|--------|-------|
| System config | ❌ | No hay UI de admin. No hay panel de configuración. |
| Database stats | ❌ | No hay vista de "cantidad de personas, fingerprints, captures". |
| Qdrant stats | ❌ | No hay vista del estado de la collección vectorial. |
| User management | ❌ | No hay CRUD de usuarios en UI. |
| Backup / restore | ❌ | No hay scripts de backup. |

---

## 12. Infraestructura y DevOps

| Feature | Estado | Notas |
|---------|--------|-------|
| Docker Compose | ✅ | Qdrant + MinIO + PostgreSQL |
| Environment config | 🟡 | `.env` con defaults. Falta doc para producción. |
| CI/CD | ❌ | No hay pipeline de CI/CD. No hay tests automatizados. |
| Monitoring | ❌ | No hay health checks, logs centralizados. |

---

## Summary: 10 features, 65 items

| Status | Count |
|--------|-------|
| ✅ Built | 26 |
| 🟡 Partial | 10 |
| ❌ Missing | 27 |
| 📋 Planned | 2 |

**Biggest gaps:**
1. ❌ **Anotación manual** (perito no puede marcar evidencia) — Plan 30-B
2. ❌ **Per-user views / RBAC** — cada usuario ve lo mismo, sin roles
3. ❌ **Case workflow** — casos sin estado, sin notas, sin annotations
4. ❌ **Administration** — sin UI de admin ni stats
5. ❌ **Search history** — el perito no puede volver a una búsqueda anterior
6. ❌ **Annotation in PDF** — el reporte no tiene evidencia del perito

**Next (already in progress):** UX de candidate list (Plan 30-A ✅ built).
**Deferred but documented:** Plan 30-B (annotation tools).
