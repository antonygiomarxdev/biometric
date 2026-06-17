# Arquitectura de Plataforma de Identidad Biométrica Unificada

## Visión General
Este documento describe la arquitectura modular diseñada para transformar el sistema de una herramienta de dactiloscopia a una plataforma de identidad biométrica multimodal escalable. El objetivo principal es permitir la integración de nuevos métodos de autenticación (facial, iris, voz) sin modificar el núcleo del sistema.

## Principios de Diseño
1.  **Desacoplamiento:** El núcleo del sistema no conoce los detalles de implementación de cada biometría.
2.  **Extensibilidad:** Nuevos métodos se añaden como "Plugins" o "Proveedores".
3.  **Abstracción:** Todos los proveedores implementan una interfaz común.

## Arquitectura del Backend

### Patrón Strategy (Proveedores)
El sistema utiliza un patrón de diseño Strategy donde `BiometricProvider` es la interfaz base.

```python
class BiometricProvider(ABC):
    @abstractmethod
    def extract_features(self, input_data: bytes) -> BiometricVector:
        pass

    @abstractmethod
    def compare(self, vector_a: BiometricVector, vector_b: BiometricVector) -> float:
        pass
```

### Estructura de Directorios
```
apps/backend/src/
├── services/
│   ├── biometrics/
│   │   ├── base.py          # Clase abstracta BiometricProvider
│   │   ├── factory.py       # Factory para instanciar proveedores
│   │   ├── providers/
│   │   │   ├── fingerprint.py  # Implementación actual
│   │   │   ├── face.py         # Implementación futura
│   │   │   └── iris.py         # Futuro
```

### Flujo de Datos
1.  **API Gateway:** Recibe `POST /api/v1/biometrics/{modality}/identify`.
2.  **BiometricFactory:** Selecciona el proveedor basado en `{modality}`.
3.  **Provider:** Procesa la entrada (imagen/audio) y extrae características.
4.  **Matcher:** Compara las características con la base de datos vectorial (PGVector).

## Escalabilidad y Procesamiento Distribuido

Para manejar altos volúmenes de usuarios y algoritmos de IA pesados (especialmente faciales), se propone:

1.  **Cola de Tareas (Celery + Redis):**
    *   La extracción de características (embedding) se mueve a workers asíncronos.
    *   La API responde inmediatamente con un `task_id` (polling o webhook).

2.  **Base de Datos Vectorial:**
    *   Uso de `Qdrant` para búsquedas de similitud eficientes a gran escala.
    *   Índices HNSW para tiempos de búsqueda sub-lineales.

## API Estandarizada
Todos los módulos biométricos deben adherirse al contrato de API:

*   **Entrada:** `multipart/form-data` con archivo y metadatos.
*   **Salida:** JSON estandarizado con `score` (0.0 - 1.0), `match` (bool) y `metadata`.

---
**Estado Actual:** Migración de módulo de huellas a arquitectura de proveedores.
