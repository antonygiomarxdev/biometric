---
spike: 002
name: llm-forensic-security
type: standard
validates: "Estrategias de seguridad, privacidad y enrutamiento multi-proveedor para LLMs en el entorno criminalístico."
verdict: VALIDATED
tags: [security, privacy, llm, gateway, legal]
---

# Spike 002: Seguridad y Privacidad de LLMs en Entornos Forenses

## 1. El Problema (Contexto Legal y Forense)
En la criminalística, los datos procesados (nombres, números de identificación, descripciones de la escena del crimen) son **Evidencia Legal Sensible**. 
- **Riesgo 1 (Data Leakage):** Si se usan APIs públicas (OpenAI estándar, Anthropic, OpenRouter), los datos pueden ser retenidos por hasta 30 días para "revisión de abuso" o usados para entrenar modelos futuros.
- **Riesgo 2 (Cadena de Custodia):** La intervención de terceros no auditables (proxies públicos) invalida la cadena de custodia de la evidencia digital.
- **Riesgo 3 (Vendor Lock-in):** Los modelos evolucionan rápido; atarse a un solo proveedor en on-premise o cloud es un riesgo operativo.

## 2. Estrategia Multi-Provider Segura

Para tener la flexibilidad de OpenRouter pero sin el riesgo de fuga de datos, la solución estándar de la industria es un **Self-Hosted AI Gateway**.

### La Herramienta Recomendada: LiteLLM Proxy (Local)
[LiteLLM](https://docs.litellm.ai/) es un proxy open-source que se despliega localmente en el laboratorio (on-premise).
- **Cómo funciona:** La aplicación (LlamaIndex) habla con LiteLLM como si fuera la API de OpenAI. LiteLLM se encarga de enrutar la petición al modelo real (Ollama, Azure, AWS Bedrock, etc.).
- **Privacidad:** Las llaves API y las reglas de enrutamiento nunca salen del servidor del laboratorio.
- **Auditoría:** Permite loggear cada prompt y respuesta en una base de datos local para propósitos de auditoría legal (esencial para peritajes).

## 3. Tiers de Privacidad para el Laboratorio

El `LLMFactory` debe soportar 3 Tiers (Niveles) de privacidad, configurables por el administrador del laboratorio:

| Nivel de Seguridad | Entorno | Proveedor Típico | Retención de Datos | Uso de Red |
|-------------------|---------|------------------|--------------------|------------|
| **Tier 1 (Máximo)** | Air-gapped (Local) | Ollama / vLLM | 0% (Todo queda en el servidor local) | Tráfico 100% interno (LAN) |
| **Tier 2 (Enterprise)** | Nube Privada | Azure OpenAI / AWS Bedrock | 0% (Por contrato BAA/Enterprise, no entrenan modelos, no revisan logs) | Tráfico cifrado (VPC/TLS) |
| **Tier 3 (Prohibido)** | API Pública | OpenAI / OpenRouter.ai | Hasta 30 días, posible uso para entrenamiento | Internet público |

## 4. Mitigación: Data Sanitization (PII Scrubbing)
Para casos donde el laboratorio decida usar Tier 2, se recomienda implementar una capa de sanitización antes del LLM.
- **Herramienta:** Microsoft Presidio (Analizador local de NLP).
- **Flujo:** 
  1. El sistema detecta nombres (ej. "Juan Pérez") y números de caso.
  2. Los enmascara: `[PERSON_1] estuvo en [LOCATION_1]`.
  3. Envía el texto enmascarado al LLM.
  4. Recibe la respuesta y des-enmascara los datos localmente.

## 5. Conclusión y Veredicto para la Arquitectura
**VEREDICTO:** Modificar la arquitectura del `LLMFactory`.
1. **Unificar el proveedor:** Eliminar múltiples proveedores (OpenAIProvider, etc.) y usar un único `LiteLLMProvider` o `OpenAICompatibleProvider`.
2. **Gateway Local:** Apuntar ese proveedor a la URL del proxy local (o directamente a Ollama/Azure según configuración).
3. Añadir documentación de seguridad en el `AI-SPEC.md`.
