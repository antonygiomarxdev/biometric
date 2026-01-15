"""Benchmark: Comparación de rendimiento CPU vs GPU."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from src.core.gpu_utils import GPUConfig, print_gpu_status
from src.services.fingerprint_service import fingerprint_service
from src.core.metrics import metrics


def load_sample_image(dataset_path: str = "data/SOCOFing/Real") -> np.ndarray:
    """Carga una imagen de muestra."""
    images = list(Path(dataset_path).glob("*.BMP"))
    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en {dataset_path}")
    
    img_path = str(images[0])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    
    return img


def benchmark_single_image(img: np.ndarray, use_gpu: bool, iterations: int = 5) -> dict:
    """Benchmark de procesamiento de una sola imagen."""
    
    # Configurar GPU
    if use_gpu:
        GPUConfig.enable()
    else:
        GPUConfig.disable()
    
    # Reset metrics
    metrics.reset()
    
    times = []
    
    print(f"\n{'='*70}")
    print(f"🔥 Benchmark: {'GPU' if use_gpu else 'CPU'}")
    print(f"{'='*70}")
    print(f"Iteraciones: {iterations}")
    print(f"Tamaño imagen: {img.shape}")
    print()
    
    for i in range(iterations):
        start = time.time()
        
        try:
            fingerprint = fingerprint_service.process_image(img, resize=True)
            elapsed = time.time() - start
            times.append(elapsed)
            
            print(f"  Iteración {i+1}/{iterations}: {elapsed:.2f}s - Minutiae: {len(fingerprint.minutiae)}")
            
        except Exception as e:
            print(f"  ❌ Error en iteración {i+1}: {e}")
            continue
    
    if not times:
        return {
            "backend": "GPU" if use_gpu else "CPU",
            "success": False,
            "error": "Todas las iteraciones fallaron"
        }
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Resultados:")
    print(f"  Promedio: {avg_time:.2f}s")
    print(f"  Mínimo:   {min_time:.2f}s")
    print(f"  Máximo:   {max_time:.2f}s")
    
    # Mostrar métricas detalladas
    print(f"\n📈 Métricas por etapa:")
    for name, data in metrics.get_stats().items():
        print(f"  {name:20s}: {data['avg']:.4f}s (llamadas: {data['count']})")
    
    return {
        "backend": "GPU" if use_gpu else "CPU",
        "success": True,
        "iterations": len(times),
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "times": times,
        "metrics": metrics.get_stats()
    }


def run_full_benchmark(dataset_path: str = "data/SOCOFing/Real", iterations: int = 5):
    """Ejecuta benchmark completo CPU vs GPU."""
    
    print("\n" + "="*70)
    print("🚀 BENCHMARK: CPU vs GPU")
    print("="*70)
    
    # Mostrar info GPU
    print_gpu_status()
    
    # Cargar imagen de muestra
    try:
        img = load_sample_image(dataset_path)
        print(f"✅ Imagen cargada: {img.shape}")
    except Exception as e:
        print(f"❌ Error cargando imagen: {e}")
        return
    
    # Benchmark CPU
    cpu_results = benchmark_single_image(img, use_gpu=False, iterations=iterations)
    
    # Benchmark GPU (si está disponible)
    if GPUConfig.get_device_info()["available"]:
        gpu_results = benchmark_single_image(img, use_gpu=True, iterations=iterations)
    else:
        print("\n⚠️ GPU no disponible, saltando benchmark GPU")
        gpu_results = None
    
    # Comparación
    print("\n" + "="*70)
    print("📊 COMPARACIÓN FINAL")
    print("="*70)
    
    if cpu_results["success"]:
        print(f"\n✅ CPU:")
        print(f"   Tiempo promedio: {cpu_results['avg_time']:.2f}s")
    else:
        print(f"\n❌ CPU: {cpu_results.get('error', 'Error desconocido')}")
    
    if gpu_results and gpu_results["success"]:
        print(f"\n✅ GPU:")
        print(f"   Tiempo promedio: {gpu_results['avg_time']:.2f}s")
        
        if cpu_results["success"]:
            speedup = cpu_results['avg_time'] / gpu_results['avg_time']
            print(f"\n🔥 Speedup: {speedup:.2f}x")
            
            if speedup > 1.5:
                print(f"   ✅ ¡GPU es {speedup:.1f}x más rápida!")
            elif speedup > 1.0:
                print(f"   ✓ GPU es ligeramente más rápida ({speedup:.1f}x)")
            else:
                print(f"   ⚠️ CPU es más rápida (speedup: {speedup:.2f}x)")
                print(f"      Esto puede deberse a overhead de transferencia GPU")
    
    elif gpu_results:
        print(f"\n❌ GPU: {gpu_results.get('error', 'Error desconocido')}")
    
    print("\n" + "="*70)
    
    # Comparación detallada por etapa
    if cpu_results["success"] and gpu_results and gpu_results["success"]:
        print("\n📈 Comparación por etapa:")
        print(f"{'Etapa':<25} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10}")
        print("-" * 70)
        
        for stage_name in cpu_results["metrics"].keys():
            cpu_time = cpu_results["metrics"][stage_name]["avg"]
            gpu_time = gpu_results["metrics"].get(stage_name, {}).get("avg", 0)
            
            if gpu_time > 0:
                stage_speedup = cpu_time / gpu_time
                speedup_str = f"{stage_speedup:.2f}x"
            else:
                speedup_str = "N/A"
            
            print(f"{stage_name:<25} {cpu_time:<12.4f} {gpu_time:<12.4f} {speedup_str:<10}")
    
    print("\n" + "="*70)


def benchmark_batch(
    dataset_path: str = "data/SOCOFing/Real",
    num_images: int = 10,
    use_gpu: bool = True
):
    """Benchmark con múltiples imágenes (simula carga real)."""
    
    print("\n" + "="*70)
    print(f"🚀 BENCHMARK BATCH: {num_images} imágenes")
    print("="*70)
    
    if use_gpu:
        GPUConfig.enable()
    else:
        GPUConfig.disable()
    
    print(f"Backend: {GPUConfig.get_backend()}")
    
    # Cargar imágenes
    images_paths = list(Path(dataset_path).glob("*.BMP"))[:num_images]
    
    if len(images_paths) == 0:
        print(f"❌ No se encontraron imágenes en {dataset_path}")
        return
    
    print(f"✅ Cargando {len(images_paths)} imágenes...")
    
    images = []
    for img_path in images_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    
    print(f"✅ {len(images)} imágenes cargadas")
    
    # Reset metrics
    metrics.reset()
    
    # Procesar todas las imágenes
    start = time.time()
    
    total_minutiae = 0
    for i, img in enumerate(images):
        try:
            fingerprint = fingerprint_service.process_image(img, resize=True)
            total_minutiae += len(fingerprint.minutiae)
            print(f"  [{i+1}/{len(images)}] Procesada - Minutiae: {len(fingerprint.minutiae)}")
        except Exception as e:
            print(f"  [{i+1}/{len(images)}] ❌ Error: {e}")
    
    total_time = time.time() - start
    
    print(f"\n📊 Resultados:")
    print(f"  Imágenes procesadas: {len(images)}")
    print(f"  Tiempo total:        {total_time:.2f}s")
    print(f"  Tiempo promedio:     {total_time / len(images):.2f}s/imagen")
    print(f"  Throughput:          {len(images) / total_time:.2f} imágenes/s")
    print(f"  Total minutiae:      {total_minutiae}")
    print(f"  Promedio minutiae:   {total_minutiae / len(images):.1f}")
    
    print(f"\n📈 Métricas por etapa:")
    for name, data in metrics.get_stats().items():
        print(f"  {name:20s}: {data['avg']:.4f}s/llamada (total: {data['total']:.2f}s)")
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "compare"],
        default="compare",
        help="Modo de benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="data/SOCOFing/Real",
        help="Path al dataset"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Número de iteraciones para benchmark single/compare"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Número de imágenes para benchmark batch"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Usar GPU para benchmark batch"
    )
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        run_full_benchmark(args.dataset, args.iterations)
    elif args.mode == "single":
        img = load_sample_image(args.dataset)
        use_gpu = GPUConfig.get_device_info()["available"] and args.gpu
        benchmark_single_image(img, use_gpu, args.iterations)
    elif args.mode == "batch":
        benchmark_batch(args.dataset, args.num_images, args.gpu)
