# ADR 003: Patrón Strategy para Proveedores Biométricos

**Estado:** Aceptado
**Fecha:** 2025-06-12
**Contexto:** El sistema debe soportar múltiples modalidades biométricas (huella, facial, iris) sin modificar el núcleo.

**Decisión:** Strategy Pattern via `BiometricProvider` ABC con `BiometricFactory` para registro y selección de proveedores.

**Estructura:**
```
services/biometrics/
├── base.py           # BiometricProvider (ABC)
├── factory.py        # BiometricFactory (registry)
└── providers/
    ├── fingerprint.py # Implementación actual
    ├── face.py        # Stub (futuro)
    └── iris.py        # Stub (futuro)
```

**Racional:**
- Open/Closed Principle: nuevas modalidades sin tocar código existente
- Factory desacopla selección de implementación
- Cada proveedor gestiona su propio pipeline de procesamiento

**Consecuencias:**
- Interfaz común puede ser restrictiva para modalidades muy diferentes
- face.py actual es stub (no implementado) — riesgo de promesa falsa
- Matching multimodal requiere lógica adicional fuera del Strategy
