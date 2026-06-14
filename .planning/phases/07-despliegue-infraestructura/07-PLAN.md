---
phase: 07-despliegue-infraestructura
plan: 01
type: execute
wave: 1
---

<objective>
Arreglar el stack de Docker y despliegue para que refleje la arquitectura actual.
</objective>

<tasks>

<task type="auto">
  <name>Task 1: Fix Backend Docker references</name>
  <action>
    El Dockerfile y docker-compose apuntan a src.api.rest:app que fue eliminado.
    - Backend Dockerfile: cambiar CMD a src.main:app
    - docker-compose.yml backend command: cambiar a src.main:app
    - docker-compose.prod.yml api command: cambiar a src.main:app
    - Agregar dependencias AI y de compliance como extras.
  </action>
</task>

<task type="auto">
  <name>Task 2: Crear .env.example</name>
  <action>
    Crear .env.example con todas las variables de entorno:
    DATABASE_URL, LLM_MODEL_NAME, LLM_API_BASE, COMPLIANCE_STRATEGY,
    JURISDICTION_COUNTRY, JURISDICTION_EXPERT_TITLE, etc.
  </action>
</task>

<task type="auto">
  <name>Task 3: Sync Makefile con comandos reales</name>
  <action>
    Actualizar Makefile para que los comandos funcionen con la nueva arquitectura.
    Asegurar que test, coverage, y docker-up funcionen.
  </action>
</task>

<task type="auto">
  <name>Task 4: Agregar docker-compose override para AI</name>
  <action>
    Si el laboratorio tiene GPU, crear docker-compose.gpu.yml que añada 
    device: nvidia/gpu al backend y configure Ollama/vLLM como servicio.
  </action>
</task>

</tasks>
