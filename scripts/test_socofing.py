"""Script simple para probar SOCOFing: indexar y buscar."""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Setup
sys.path.insert(0, str(Path(__file__).parent))

from src.services.fingerprint_service import fingerprint_service
from src.services.comparison_service import comparison_service
from src.storage.database import db_manager
from src.core.metrics import metrics

# Configuración
DATASET_PATH = "data/SOCOFing/Real"  # Cambia esto si está en otra ubicación
MAX_REGISTER = 100  # Número máximo de huellas a registrar (None = todas)


def setup_db():
    """Inicializa la base de datos."""
    print("🔧 Inicializando base de datos...")
    db_manager.create_tables()
    print("✅ DB lista\n")


def parse_filename(filename):
    """Extrae info del nombre: 1__M_Left_index_finger.BMP"""
    parts = filename.replace(".BMP", "").split("__")
    person_id = f"P{parts[0].zfill(4)}"
    info = parts[1] if len(parts) > 1 else "Unknown"
    return person_id, info


def index_dataset():
    """Indexa todas las huellas del dataset."""
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        print(f"❌ No se encuentra el dataset en: {dataset_path}")
        print("   Ajusta DATASET_PATH en el script")
        return False
    
    images = list(dataset_path.glob("*.BMP"))
    
    if MAX_REGISTER:
        images = images[:MAX_REGISTER]
    
    print(f"📁 Encontradas {len(images)} imágenes en {DATASET_PATH}")
    print(f"🚀 Iniciando indexación...\n")
    
    # Cargar todas las imágenes primero (más eficiente)
    print("📦 Cargando imágenes en memoria...")
    loaded_images = []
    loaded_paths = []
    for img_path in tqdm(images, desc="Cargando"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            loaded_images.append(img)
            loaded_paths.append(img_path)
    
    print(f"✅ {len(loaded_images)} imágenes cargadas\n")
    
    start = time.time()
    success = 0
    errors = 0
    
    # Procesar imágenes
    for img, img_path in tqdm(zip(loaded_images, loaded_paths), total=len(loaded_images), desc="Procesando"):
        try:
            # Extraer features
            fingerprint = fingerprint_service.process_image(
                img,
                fingerprint_id=img_path.name,
                resize=True
            )
            
            if not fingerprint.minutiae:
                errors += 1
                continue
            
            # Registrar
            person_id, info = parse_filename(img_path.name)
            comparison_service.register_fingerprint(
                fingerprint=fingerprint,
                person_id=person_id,
                name=f"Person {person_id} - {info}",
                document=person_id
            )
            
            success += 1
            
        except Exception as e:
            errors += 1
            if errors <= 5:  # Mostrar solo los primeros 5 errores
                print(f"\n❌ Error en {img_path.name}: {type(e).__name__}: {e}")
    
    duration = time.time() - start
    
    print(f"\n✅ Indexación completada!")
    print(f"   - Exitosas: {success}")
    print(f"   - Errores: {errors}")
    
    if success > 0:
        print(f"   - Tiempo: {duration:.1f}s ({duration/success:.2f}s por huella)")
    else:
        print(f"   - Tiempo: {duration:.1f}s")
        print(f"\n⚠️  Todas las imágenes fallaron. Revisa los errores arriba.")
    
    return success > 0


def search_fingerprint(image_path):
    """Busca una huella en el índice."""
    print(f"\n🔍 Buscando: {image_path}")
    
    # Leer imagen
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ No se pudo leer la imagen")
        return
    
    # Extraer features
    print("   Extrayendo features...")
    fingerprint = fingerprint_service.process_image(img)
    
    if not fingerprint.minutiae:
        print("   ❌ No se extrajeron minutiae")
        return
    
    print(f"   ✅ Extraídas {len(fingerprint.minutiae)} minutiae")
    
    # Buscar
    print("   Buscando en el índice...")
    result = comparison_service.identify(fingerprint)
    
    print("\n" + "="*60)
    if result.matched:
        print("✅ MATCH ENCONTRADO!")
        print(f"   Persona: {result.name}")
        print(f"   ID: {result.person_id}")
        print(f"   Documento: {result.document}")
        print(f"\n📊 Métricas:")
        print(f"   Confidence: {result.score:.1%} ({result.score:.4f} de 1.0)")
        print(f"   Distancia: {result.distance:.2f} (menor = más similar)")
        
        # Interpretar confidence
        if result.score > 0.9:
            print(f"   ⭐ Confianza: MUY ALTA (casi seguro)")
        elif result.score > 0.7:
            print(f"   ✓ Confianza: ALTA (probable match)")
        elif result.score > 0.5:
            print(f"   ⚠️  Confianza: MEDIA (revisar manualmente)")
        else:
            print(f"   ⚠️  Confianza: BAJA (puede ser falso positivo)")
    else:
        print("❌ NO SE ENCONTRÓ MATCH")
        if result.person_id:
            print(f"\n📊 Mejor candidato encontrado:")
            print(f"   Persona: {result.name}")
            print(f"   ID: {result.person_id}")
            print(f"   Confidence: {result.score:.1%} ({result.score:.4f} de 1.0)")
            print(f"   Distancia: {result.distance:.2f}")
            print(f"\n💡 Explicación:")
            print(f"   - Confidence: {result.score:.1%} = probabilidad de que sea la misma persona")
            print(f"   - Distancia: {result.distance:.2f} (umbral: 2000.0)")
            print(f"   - La distancia es mayor al umbral, por eso NO es match")
        else:
            print(f"   No se encontraron candidatos en la base de datos")
    print("="*60)


def interactive_search():
    """Modo interactivo para buscar huellas."""
    print("\n🔍 MODO BÚSQUEDA INTERACTIVA")
    print("="*60)
    
    dataset_path = Path(DATASET_PATH)
    images = list(dataset_path.glob("*.BMP"))
    
    while True:
        print(f"\nOpciones:")
        print("  1. Buscar imagen aleatoria del dataset")
        print("  2. Buscar imagen específica (ruta)")
        print("  3. Ver estadísticas")
        print("  4. Salir")
        
        choice = input("\nElige opción: ").strip()
        
        if choice == "1":
            # Buscar imagen aleatoria
            img = np.random.choice(images)
            search_fingerprint(str(img))
            
        elif choice == "2":
            # Buscar imagen específica
            path = input("Ruta de la imagen: ").strip()
            if os.path.exists(path):
                search_fingerprint(path)
            else:
                print("❌ Archivo no encontrado")
        
        elif choice == "3":
            # Estadísticas
            from src.storage.repository import repository
            from src.storage.vector_index import vector_index
            
            print("\n📊 ESTADÍSTICAS")
            print(f"   Huellas en DB: {repository.count()}")
            print(f"   Vectores en índice: {vector_index.size()}")
            
            stats = metrics.get_stats("repository_identify")
            if stats:
                print(f"\n   Búsquedas realizadas: {stats['count']}")
                print(f"   Latencia promedio: {stats['mean']:.2f}ms")
                print(f"   P50/P95: {stats['p50']:.2f}ms / {stats['p95']:.2f}ms")
        
        elif choice == "4":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")


def main():
    """Función principal."""
    from src.core.gpu_utils import GPUConfig
    
    # TEMPORAL: Deshabilitar GPU porque el código actual tiene mucho overhead
    # TODO: Optimizar código GPU para mantener datos en GPU durante todo el pipeline
    # GPUConfig.disable() # Comentado para usar la nueva lógica optimizada
    
    print("="*60)
    print("🔬 TEST SOCOFING - Sistema Biométrico")
    print("="*60)
    print(f"⚙️  Backend: {GPUConfig.get_backend()}")
    print("="*60)
    
    # Setup inicial
    setup_db()
    
    # Menú
    print("¿Qué quieres hacer?")
    print("  1. Indexar dataset (extraer features y guardar)")
    print("  2. Buscar huella (asume que ya indexaste)")
    print("  3. Indexar Y buscar (flujo completo)")
    
    choice = input("\nElige opción [3]: ").strip() or "3"
    
    if choice == "1":
        # Solo indexar
        if index_dataset():
            print("\n✅ Listo! Ahora puedes buscar huellas")
            print("   Ejecuta de nuevo y elige opción 2")
    
    elif choice == "2":
        # Solo buscar
        interactive_search()
    
    elif choice == "3":
        # Flujo completo
        if index_dataset():
            print("\n✅ Indexación completa!")
            print("   Ahora vamos a buscar...\n")
            input("Presiona Enter para continuar...")
            interactive_search()
    
    else:
        print("❌ Opción inválida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
