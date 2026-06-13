# ADR 004: Cadena de Custodia con Audit Trail Append-Only

**Estado:** Propuesto (para Fase 2)
**Fecha:** 2025-06-12
**Contexto:** Requisito forense: cada operación sobre huellas debe ser trazable con integridad criptográfica.

**Decisión:** Tabla `audit_logs` con:
- Registro append-only (nunca se eliminan filas)
- Hash chain: cada fila contiene SHA-256 de la fila anterior
- Firma digital opcional (RSA) para no-repudio
- Timestamp NTP sincronizado
- IP, user-agent, operator_id en cada registro

**Estructura:**
```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    operator_id UUID,
    ip_address INET,
    details JSONB,
    previous_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);
```

**Racional:**
- Append-only garantiza integridad (no se puede modificar historial)
- Hash chain permite verificar que ningún registro fue alterado
- Firma digital añade no-repudio para uso judicial
- JSONB en details permite flexibilidad sin migraciones de esquema

**Consecuencias:**
- Crecimiento de tabla lineal con operaciones (requiere partitioning)
- Costo computacional de hash chain es mínimo (SHA-256)
- Firma digital añade latencia (evaluar si es necesaria para todos los eventos)
- Partitioning por mes para mantener rendimiento
