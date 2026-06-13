---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: "Fase 1: El Flujo Core Forense (MVP Vertical)"
status: 🟢 En Ejecución
last_updated: "2026-06-13T05:47:47.169Z"
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# State: Biometric

**Last updated:** 2025-06-12
**Current phase:** Fase 1: El Flujo Core Forense (MVP Vertical)
**Status:** 🟢 En Ejecución

## Project Reference

See: `.planning/PROJECT.md`
**Core value:** Darle al perito forense una herramienta digital para identificar huellas latentes más rápido y con trazabilidad, manteniendo su criterio como autoridad final.
**Competitive Advantage:** Doble motor de Inteligencia Artificial (Visión Computacional para el procesamiento de la imagen + GenAI para la automatización de la burocracia judicial).

## Phase Progress

| Phase | Status | Progress |
|-------|--------|----------|
| 1. Flujo Core Forense | 🏃‍♂️ Empezando | 0% |
| 2. IA Visión Computacional | ⏳ Pendiente | 0% |
| 3. IA Generativa (Dictámenes) | ⏳ Pendiente | 0% |
| 4. Despliegue On-Premise | ⏳ Pendiente | 0% |

## Current Work

Se ha redefinido el Roadmap para centrarse en entregas verticales (vertical slices). Se ha adoptado formalmente la estrategia de "Doble Motor de IA" como ventaja competitiva. Ahora comenzamos la Fase 1: implementar el flujo básico end-to-end para que el perito pueda subir, buscar y comparar visualmente.

## Next Actions

1. Diseñar el esquema de base de datos para la ingesta (PostgreSQL + pgvector).
2. Refactorizar/dividir `rest.py` para soportar el flujo core (Upload, Process, Search, Audit).
3. Desarrollar la UI básica en React (Pantalla de carga, Lista de Candidatos, Visor Lado a Lado).
