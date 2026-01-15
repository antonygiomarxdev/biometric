"""
Módulo de Normalización y Consenso.
Garantiza la invariancia y limpieza de los datos antes de la vectorización.
"""
from typing import List, Tuple, Dict
import numpy as np
from dataclasses import replace

from src.core.types import MinutiaCandidate, NormalizedFingerprint, AlgorithmOrigin, MinutiaType
from src.core.interfaces import INormalizer

class MinutiaNormalizer(INormalizer):
    """
    Implementación estándar de normalización geométrica.
    """
    
    def normalize(self, minutiae: List[MinutiaCandidate], img_shape: Tuple[int, int]) -> NormalizedFingerprint:
        if not minutiae:
            return NormalizedFingerprint(id="unknown", minutiae=[], width=img_shape[1], height=img_shape[0])
            
        # 1. Consenso (eliminar duplicados cercanos)
        unique_minutiae = self._apply_consensus(minutiae)
        
        # 2. Centrado
        centered_minutiae = self._center_minutiae(unique_minutiae, img_shape)
        
        # 3. Ordenamiento Canónico
        sorted_minutiae = self._canonical_sort(centered_minutiae)
        
        return NormalizedFingerprint(
            id="unknown", # Se asignará fuera
            minutiae=sorted_minutiae,
            width=img_shape[1],
            height=img_shape[0]
        )
    
    def _apply_consensus(self, candidates: List[MinutiaCandidate], distance_thresh: int = 5) -> List[MinutiaCandidate]:
        """Fusiona candidatos que están muy cerca."""
        if not candidates:
            return []
            
        # Ordenar por confianza descendente para priorizar los mejores
        candidates = sorted(candidates, key=lambda m: m.confidence, reverse=True)
        
        final_list: List[MinutiaCandidate] = []
        
        for cand in candidates:
            is_duplicate = False
            for existing in final_list:
                dist = np.sqrt((cand.x - existing.x)**2 + (cand.y - existing.y)**2)
                if dist < distance_thresh:
                    # Ya existe uno mejor (mayor confianza) cerca
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_list.append(cand)
                
        return final_list

    def _center_minutiae(self, minutiae: List[MinutiaCandidate], img_shape: Tuple[int, int]) -> List[MinutiaCandidate]:
        """Centra las minucias restando el centroide."""
        if not minutiae:
            return []
            
        xs = [m.x for m in minutiae]
        ys = [m.y for m in minutiae]
        
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # Opcional: Centrar al centro de la imagen si se prefiere absoluto
        # cx, cy = img_shape[1] / 2, img_shape[0] / 2
        
        centered = []
        for m in minutiae:
            # Shift y asegurar enteros? No, float para precisión en vector
            # Pero MinutiaCandidate define x,y como int.
            # Mantengamos int para compatibilidad, pero relativos
            new_x = int(m.x - cx)
            new_y = int(m.y - cy)
            centered.append(replace(m, x=new_x, y=new_y))
            
        return centered

    def _canonical_sort(self, minutiae: List[MinutiaCandidate]) -> List[MinutiaCandidate]:
        """
        Ordena las minucias de forma determinista.
        Criterio: 
        1. Distancia al origen (radio)
        2. Ángulo polar
        """
        def sort_key(m: MinutiaCandidate):
            r = m.x**2 + m.y**2
            # theta = np.arctan2(m.y, m.x) 
            # Usamos coordenadas directamente para estabilidad si r es igual
            return (r, m.y, m.x, m.angle)
            
        return sorted(minutiae, key=sort_key)

class RotationInvariantNormalizer(MinutiaNormalizer):
    """
    Alinea la huella basándose en la orientación dominante (PCA de coordenadas de minucias).
    """
    def normalize(self, minutiae: List[MinutiaCandidate], img_shape: Tuple[int, int]) -> NormalizedFingerprint:
        if len(minutiae) < 3:
            return super().normalize(minutiae, img_shape)
            
        # 1. Centrar (super)
        # Necesitamos llamar a los métodos protegidos o reusar logica
        # Mejor heredar y extender _center_minutiae o hacerlo en pipeline
        
        # Copiar lógica base para tener control
        unique = self._apply_consensus(minutiae)
        centered = self._center_minutiae(unique, img_shape)
        
        # 2. Rotar usando PCA de las posiciones (x, y)
        coords = np.array([[m.x, m.y] for m in centered])
        if len(coords) > 2:
            # Covarianza
            cov = np.cov(coords.T)
            vals, vecs = np.linalg.eigh(cov)
            
            # Vector propio principal (mayor autovalor)
            # vecs[:, 1] es el vector principal (eigh ordena ascendente)
            major_axis = vecs[:, 1]
            angle_rad = np.arctan2(major_axis[1], major_axis[0])
            
            # Rotar para que el eje principal sea vertical (90 grados, pi/2)
            # Queremos rotar -angle + pi/2
            rotation_angle = -angle_rad + np.pi/2
            
            rotated = []
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            
            for m in centered:
                # Rotar coordenadas
                rx = m.x * cos_a - m.y * sin_a
                ry = m.x * sin_a + m.y * cos_a
                
                # Rotar orientación de la minucia
                ra = m.angle + np.degrees(rotation_angle)
                ra = ra % 360
                
                rotated.append(replace(m, x=int(rx), y=int(ry), angle=ra))
            
            centered = rotated

        # 3. Ordenamiento Canónico
        sorted_minutiae = self._canonical_sort(centered)
        
        return NormalizedFingerprint(
            id="unknown",
            minutiae=sorted_minutiae,
            width=img_shape[1],
            height=img_shape[0]
        )
