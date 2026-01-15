"""Script simple: procesar UNA imagen y buscar en la base de datos."""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.services.fingerprint_service import fingerprint_service
from src.services.comparison_service import comparison_service
from src.storage.repository import repository
from src.core.metrics import metrics
from src.core.gpu_utils import GPUConfig
from src.core.config import config


def identify_image(image_path: str):
    """Procesa una imagen y busca coincidencias en la base de datos."""
    
    print("="*70)
    print("🔍 IDENTIFICACIÓN DE HUELLA DACTILAR")
    print("="*70)
    print(f"⚙️  Backend: {GPUConfig.get_backend()}")
    print("="*70)
    print(f"\n📸 Imagen: {image_path}\n")
    
    # 1. Procesar imagen
    print("1️⃣ Extrayendo features de la imagen...")
    start = time.time()
    
    try:
        fingerprint = fingerprint_service.process_image_from_path(image_path)
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")
        return
    
    process_time = time.time() - start
    
    if not fingerprint.minutiae:
        print("❌ No se pudieron extraer minutiae de la imagen")
        print("   Posibles causas:")
        print("   - Imagen de baja calidad")
        print("   - No es una huella dactilar")
        print("   - Formato incorrecto")
        return
    
    print(f"✅ Minutiae extraídas: {len(fingerprint.minutiae)}")
    print(f"   - Terminaciones: {sum(1 for m in fingerprint.minutiae if m.type == 'termination')}")
    print(f"   - Bifurcaciones: {sum(1 for m in fingerprint.minutiae if m.type == 'bifurcation')}")
    print(f"   - Tiempo: {process_time:.2f}s")
    
    # 2. Buscar en la base de datos
    print("\n2️⃣ Buscando en la base de datos...")
    
    # Verificar que hay huellas indexadas
    total_indexed = repository.count()
    if total_indexed == 0:
        print("❌ La base de datos está vacía")
        print("   Primero debes indexar huellas con: python test_socofing.py")
        return
    
    print(f"   Huellas indexadas: {total_indexed}")
    
    start = time.time()
    result = comparison_service.identify(fingerprint)
    search_time = time.time() - start
    
    print(f"   Tiempo de búsqueda: {search_time*1000:.2f}ms")
    
    # 3. Mostrar resultado
    print("\n" + "="*70)
    
    if result.matched:
        print("✅ ¡COINCIDENCIA ENCONTRADA!")
        print("="*70)
        print(f"\n👤 Persona Identificada:")
        print(f"   Nombre:    {result.name}")
        print(f"   ID:        {result.person_id}")
        print(f"   Documento: {result.document}")
        print(f"\n📊 Métricas de Match:")
        print(f"   Confidence: {result.score:.1%} ({result.score:.4f} de 1.0)")
        print(f"   Distancia: {result.distance:.2f} (menor = más similar)")
        
        # Interpretar el resultado
        if result.score > 0.9:
            confidence = "MUY ALTA ✅ (casi seguro)"
        elif result.score > 0.7:
            confidence = "ALTA ✓ (probable match)"
        elif result.score > 0.5:
            confidence = "MEDIA ⚠️ (revisar manualmente)"
        else:
            confidence = "BAJA ⚠️ (puede ser falso positivo)"
        
        print(f"   Confianza: {confidence}")
        print(f"\n💡 Explicación:")
        print(f"   - Confidence (0-100%): Probabilidad de que sea la misma persona")
        print(f"     • 90-100% = Muy alta confianza")
        print(f"     • 70-90% = Alta confianza")
        print(f"     • 50-70% = Confianza media")
        print(f"     • <50% = Baja confianza")
        print(f"   - Distancia: Diferencia entre vectores (0 = idéntico, mayor = más diferente)")
        
    else:
        print("❌ NO SE ENCONTRÓ COINCIDENCIA")
        print("="*70)
        
        if result.person_id:
            print(f"\n📊 Mejor candidato encontrado:")
            print(f"   Persona: {result.name}")
            print(f"   ID: {result.person_id}")
            print(f"   Confidence: {result.score:.1%} ({result.score:.4f} de 1.0)")
            print(f"   Distancia: {result.distance:.2f}")
            print(f"\n💡 Explicación:")
            print(f"   - Confidence: {result.score:.1%} = probabilidad de que sea la misma persona")
            print(f"   - Distancia: {result.distance:.2f} (umbral máximo: {config.match_threshold:.0f})")
            print(f"   - La distancia es mayor al umbral, por eso NO es match")
            print(f"   - Esto significa que la huella NO coincide con ninguna registrada")
        else:
            print(f"\n📊 No se encontraron candidatos")
            print(f"   La base de datos puede estar vacía")
    
    print("\n" + "="*70)
    print("\n⏱️  Resumen de tiempos:")
    print(f"   Extracción:  {process_time:.2f}s")
    print(f"   Búsqueda:    {search_time*1000:.2f}ms")
    print(f"   Total:       {process_time + search_time:.2f}s")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python identify_single.py <ruta_imagen>")
        print("\nEjemplos:")
        print('  python identify_single.py "data/SOCOFing/Real/1__M_Left_index_finger.BMP"')
        print('  python identify_single.py "mi_huella.jpg"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ No se encuentra el archivo: {image_path}")
        sys.exit(1)
    
    identify_image(image_path)
