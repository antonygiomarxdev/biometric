# Especificaciones Técnicas: Módulo de Reconocimiento Facial

Este documento define los requisitos y especificaciones para la futura integración del módulo de reconocimiento facial en la Plataforma Biométrica Unificada.

## 1. Requisitos de Hardware (Captura)

Para garantizar la seguridad y precisión, se recomiendan los siguientes estándares de captura:

*   **Cámaras Web Estándar (Nivel Básico):**
    *   Resolución mínima: 720p (1280x720).
    *   Framerate: 30 fps.
    *   Iluminación: Controlada, evitando contraluz fuerte.
*   **Cámaras de Profundidad/IR (Nivel Avanzado - Anti-Spoofing):**
    *   Soporte para Intel RealSense o cámaras compatibles con Windows Hello.
    *   Capacidad de captura infrarroja para detección de vivacidad activa/pasiva.

## 2. Algoritmos de IA y Modelos

El módulo debe implementar una pipeline de procesamiento en tres etapas:

### Etapa 1: Detección y Alineación de Rostros
*   **Algoritmo:** RetinaFace o MTCNN.
*   **Función:** Localizar el rostro en la imagen y alinear los puntos clave (ojos, nariz, boca) para normalizar la pose.

### Etapa 2: Detección de Vivacidad (Liveness Detection)
*   **Objetivo:** Prevenir ataques de presentación (fotos, videos, máscaras).
*   **Métodos:**
    *   *Pasivo:* Análisis de textura y reflexión de luz (modelos Silent-Face-Anti-Spoofing).
    *   *Activo:* Desafío de movimiento (parpadeo, giro de cabeza) o análisis de profundidad (si hay hardware IR).

### Etapa 3: Extracción de Embeddings (Reconocimiento)
*   **Modelo:** ArcFace (ResNet-100 backbone) o FaceNet.
*   **Salida:** Vector de características de 512 dimensiones (float32).
*   **Rendimiento:** < 200ms por inferencia en CPU moderna, < 20ms en GPU.

## 3. Niveles de Precisión y Seguridad

*   **Falsos Positivos (FAR):** < 0.001% (1 en 100,000).
*   **Falsos Negativos (FRR):** < 1% en condiciones controladas.
*   **Umbral de Confianza:** Ajustable según caso de uso (Seguridad Alta vs. Conveniencia).

## 4. Protocolo de Integración

El módulo facial debe implementar la interfaz `BiometricProvider`:

```python
class FaceProvider(BiometricProvider):
    def extract(self, image):
        # 1. Detect & Align
        # 2. Liveness Check (raise Error if spoof)
        # 3. Extract Embedding
        return vector
```

## 5. Consideraciones de Privacidad
*   No almacenar imágenes crudas de rostros permanentemente a menos que sea requerido por ley.
*   Almacenar solo los vectores numéricos (embeddings) que no son reversibles a la imagen original.
