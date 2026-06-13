"""
Benchmark de paralelismo y robustez del nuevo pipeline.
"""
import sys
import os
import time
import cv2
import numpy as np
import logging
from typing import List

# Añadir root al path
sys.path.append(os.getcwd())

from src.services.fingerprint_service import fingerprint_service
# GPU support removed (CPU-only)
from src.core.types import NormalizedFingerprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

def generate_synthetic_fingerprint(size=(300, 300)) -> np.ndarray:
    """Genera una imagen de ruido que simula huella para carga de trabajo."""
    img = np.random.randint(0, 255, size, dtype=np.uint8)
    # Añadir estructura para que el enhancer trabaje algo
    for i in range(0, size[0], 10):
        cv2.line(img, (0, i), (size[1], i), (0), 2)
    return img

def run_benchmark(num_images: int = 20):
    logger.info(f"Generando {num_images} imágenes sintéticas...")
    images = [generate_synthetic_fingerprint() for _ in range(num_images)]
    ids = [f"id_{i}" for i in range(num_images)]
    
    logger.info(f"Iniciando benchmark con GPU={False}")
    
    start_time = time.time()
    results = fingerprint_service.process_batch(images, fingerprint_ids=ids)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_images
    
    valid_results = sum(1 for r in results if r is not None)
    
    print("\n" + "="*50)
    print("RESULTADOS BENCHMARK")
    print("="*50)
    print(f"Modo:         {'GPU' if False else 'CPU (Parallel)'}")
    print(f"Imágenes:     {num_images}")
    print(f"Tiempo Total: {total_time:.4f} s")
    print(f"Tiempo/Img:   {avg_time:.4f} s")
    print(f"FPS:          {1/avg_time:.2f}")
    print(f"Éxito:        {valid_results}/{num_images}")
    print("="*50 + "\n")
    
    # Verificar estructura del resultado
    if valid_results > 0:
        first: NormalizedFingerprint = results[0]
        print("Estructura de resultado (Clean Code Check):")
        print(f"ID: {first.id}")
        print(f"Minutiae Count: {len(first.minutiae)}")
        print(f"Vector Shape: {first.vector.shape}")
        
if __name__ == "__main__":
    # Forzar CPU para probar paralelismo
    # os.environ["FORCE_CPU"] = "1" 
    run_benchmark()
