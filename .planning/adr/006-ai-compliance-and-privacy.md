# ADR 006: AI Compliance, Privacy & Forensic Data Handling

**Date:** 2026-06-13
**Status:** Accepted
**Context:** The integration of LLMs (Phase 3) for generating forensic reports involves processing highly sensitive PII (Personally Identifiable Information) and digital evidence. Using third-party LLM providers poses critical legal and compliance risks.

## Relevant International Standards
If audited, the system will be evaluated against:
1. **ISO/IEC 27037:2012** (Guidelines for identification, collection, acquisition and preservation of digital evidence): Requires strict Chain of Custody. Data sent to untrusted third parties breaks this chain.
2. **NIST SP 800-53 (Rev. 5)** (Security and Privacy Controls for Information Systems): Requires explicit control over data residency and zero trust architecture for external system interconnections.
3. **GDPR / Local Data Protection Laws (e.g., Ley 896 Nicaragua)**: Explicit prohibition of cross-border transfer of sensitive biometric and criminal data without explicit consent or adequate safeguards.

## Vulnerabilities of Standard Cloud LLMs (e.g., OpenAI Public API)
- **Data Retention:** Default retention is 30 days for "trust and safety" review. This means human contractors at the AI company could theoretically read forensic reports. **(CRITICAL COMPLIANCE VIOLATION)**
- **Data Residency:** Data is processed in US servers, violating local sovereignty laws for criminal data.
- **Training Use:** Free tiers and some API terms allow data to be used for model training, creating the risk of data leakage (models reciting case details to other users).

## Decision: The "Zero-Retention / Air-Gapped" Doctrine

To pass any forensic or security audit, the architecture MUST enforce the following constraints at the infrastructure level:

### 1. Primary Operation Mode: Fully Air-Gapped (Tier 1)
- The system must be capable of running 100% disconnected from the internet.
- **Implementation:** Use local models via Ollama or vLLM running on local GPU clusters within the laboratory's DMZ.

### 2. Cloud Fallback Mode: Zero-Retention Enterprise Contracts (Tier 2)
- If local hardware is insufficient and cloud processing is mandated, the system may ONLY connect to cloud providers under strict **Enterprise Agreements (e.g., Azure OpenAI Services or AWS Bedrock)**.
- **Audit Proof:** The laboratory must possess a signed BAA (Business Associate Agreement) or explicit "Zero Data Retention" and "No Training" addendums from the cloud provider.
- Public/Standard APIs (including OpenRouter.ai standard tier) are strictly banned.

### 3. Architecture: Unified OpenAI-Compatible Interface + Gateway
- The backend will NOT hardcode specific provider SDKs (no `anthropic` or proprietary libraries).
- **Implementation:** The `LLMFactory` will exclusively use the `OpenAI-Compatible REST API` standard.
- The `api_base` will point either to:
  a) The local Ollama instance (`http://localhost:11434/v1`)
  b) A local AI Gateway (like LiteLLM Proxy) that securely routes requests within the VPC.
  c) The dedicated Azure Enterprise endpoint.

### 4. Application-Level Data Scrubbing (Future Phase)
- Before any prompt is sent to the LLM (even local), PII (names, ID numbers) should ideally be masked using a local NLP scrubber (e.g., Microsoft Presidio).

## Consequences
- **Positive:** Passes ISO 27037 and data sovereignty audits. Protects the laboratory from massive legal liabilities and invalidated court cases.
- **Negative:** Prevents the use of the absolute cutting-edge models (like Claude 3.5 Sonnet or GPT-4o) UNLESS they are deployed within an approved Enterprise VPC environment (like Azure or AWS). We are restricted to open-weights models (Llama 3, Mistral) for local deployment.
