"""Script para visualizar la detección de minutiae paso a paso."""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.services.fingerprint_service import fingerprint_service


def visualize_pipeline(image_path: str):
    """Visualiza todo el pipeline de procesamiento."""
    
    print(f"📸 Procesando: {image_path}\n")
    
    # 1. Cargar imagen original
    print("1️⃣ Cargando imagen original...")
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        print(f"❌ No se pudo cargar: {image_path}")
        return
    
    print(f"   Tamaño: {img_original.shape}")
    
    # 2. Procesar con el servicio
    print("2️⃣ Procesando imagen (enhancement + extraction)...")
    start = time.time()
    
    # Enhancement
    enhancer = fingerprint_service.enhancer
    enhanced = enhancer.enhance(img_original.copy())
    print(f"   ✅ Enhanced en {time.time()-start:.2f}s")
    
    # Skeletonization
    start = time.time()
    skeleton = enhancer.skeletonize(enhanced)
    print(f"   ✅ Skeletonized en {time.time()-start:.2f}s")
    
    # Extracción de minutiae
    start = time.time()
    extractor = fingerprint_service.extractor
    minutiae = extractor.extract(skeleton)
    print(f"   ✅ Minutiae extraídas en {time.time()-start:.2f}s")
    
    # Estadísticas
    terminations = sum(1 for m in minutiae if m.type == "termination")
    bifurcations = sum(1 for m in minutiae if m.type == "bifurcation")
    
    print(f"\n📊 RESULTADOS:")
    print(f"   Total minutiae: {len(minutiae)}")
    print(f"   - Terminaciones: {terminations} (rojo)")
    print(f"   - Bifurcaciones: {bifurcations} (azul)")
    
    # 3. Crear visualización
    print("\n3️⃣ Generando visualización...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Pipeline de Detección de Minutiae\n{Path(image_path).name}', 
                 fontsize=16, fontweight='bold')
    
    # Original
    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title('1. Imagen Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Enhanced
    axes[0, 1].imshow(enhanced, cmap='gray')
    axes[0, 1].set_title('2. Enhanced (Gabor + Ridge)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Skeleton
    axes[1, 0].imshow(skeleton, cmap='gray')
    axes[1, 0].set_title('3. Skeletonized', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Minutiae sobre skeleton
    # Convertir skeleton a RGB para dibujar en color
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    
    for m in minutiae:
        x, y = m.position
        if m.type == "termination":
            cv2.circle(skeleton_rgb, (x, y), 3, (255, 0, 0), -1)  # Rojo
            # Dibujar orientación
            length = 10
            angle_rad = np.radians(m.orientation)
            end_x = int(x + length * np.cos(angle_rad))
            end_y = int(y + length * np.sin(angle_rad))
            cv2.line(skeleton_rgb, (x, y), (end_x, end_y), (255, 0, 0), 1)
        else:  # bifurcation
            cv2.circle(skeleton_rgb, (x, y), 3, (0, 0, 255), -1)  # Azul
    
    axes[1, 1].imshow(skeleton_rgb)
    axes[1, 1].set_title(
        f'4. Minutiae Detectadas\n'
        f'Terminaciones (rojo): {terminations} | Bifurcaciones (azul): {bifurcations}',
        fontsize=12, fontweight='bold'
    )
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Guardar
    output_path = f"output_visualization_{Path(image_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Guardado en: {output_path}")
    
    # Mostrar
    print("\n👁️  Mostrando visualización (cierra la ventana para continuar)...")
    plt.show()


def batch_visualize(dataset_path: str, num_samples: int = 5):
    """Visualiza múltiples imágenes del dataset."""
    from pathlib import Path
    import random
    
    images = list(Path(dataset_path).glob("*.BMP"))
    samples = random.sample(images, min(num_samples, len(images)))
    
    print(f"📁 Visualizando {len(samples)} imágenes del dataset...\n")
    
    for i, img_path in enumerate(samples, 1):
        print(f"\n{'='*70}")
        print(f"Imagen {i}/{len(samples)}")
        print(f"{'='*70}")
        visualize_pipeline(str(img_path))
        
        if i < len(samples):
            input("\nPresiona Enter para la siguiente imagen...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualizar detección de minutiae")
    parser.add_argument("image", help="Ruta a imagen o dataset")
    parser.add_argument("-n", "--num", type=int, default=5, 
                       help="Número de imágenes a visualizar (modo batch)")
    parser.add_argument("--batch", action="store_true",
                       help="Modo batch: visualiza múltiples imágenes del dataset")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_visualize(args.image, args.num)
    else:
        visualize_pipeline(args.image)
