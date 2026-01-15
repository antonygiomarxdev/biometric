"""Script rápido para pruebas básicas con SOCOFing."""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.fingerprint_service import fingerprint_service
from src.services.comparison_service import comparison_service
from src.storage.database import db_manager

def main():
    """Test rápido: registra 2 huellas e identifica una."""
    
    print("🧪 Test Rápido con SOCOFing\n")
    
    # Inicializar DB
    print("1️⃣ Inicializando base de datos...")
    db_manager.create_tables()
    print("   ✅ DB lista\n")
    
    # Rutas de ejemplo (ajustar según tu estructura)
    base_path = Path("data/SOCOFing/Real")
    
    if not base_path.exists():
        print(f"❌ No se encuentra el dataset en: {base_path}")
        print("   Descarga SOCOFing y colócalo en data/SOCOFing/")
        return
    
    # Buscar algunas imágenes
    images = list(base_path.glob("*.BMP"))[:5]
    
    if len(images) < 2:
        print("❌ No se encontraron suficientes imágenes")
        return
    
    print(f"2️⃣ Encontradas {len(images)} imágenes para probar\n")
    
    # Registrar primera huella
    print("3️⃣ Registrando huella 1...")
    img1 = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    fp1 = fingerprint_service.process_image(img1, fingerprint_id=images[0].name)
    
    if not fp1.minutiae:
        print("   ⚠️ No se extrajeron minutiae de la imagen 1")
        return
    
    print(f"   ✅ Extraídas {len(fp1.minutiae)} minutiae")
    
    rec_id = comparison_service.register_fingerprint(
        fingerprint=fp1,
        person_id="P001",
        name="Test Person 1",
        document="12345678"
    )
    print(f"   ✅ Registrada con ID: {rec_id}\n")
    
    # Registrar segunda huella (persona diferente)
    print("4️⃣ Registrando huella 2...")
    img2 = cv2.imread(str(images[1]), cv2.IMREAD_GRAYSCALE)
    fp2 = fingerprint_service.process_image(img2, fingerprint_id=images[1].name)
    
    if fp2.minutiae:
        comparison_service.register_fingerprint(
            fingerprint=fp2,
            person_id="P002",
            name="Test Person 2",
            document="87654321"
        )
        print(f"   ✅ Extraídas {len(fp2.minutiae)} minutiae\n")
    
    # Identificar la primera huella
    print("5️⃣ Identificando huella 1 nuevamente...")
    result = comparison_service.identify(fp1)
    
    if result.matched:
        print(f"   ✅ ¡MATCH ENCONTRADO!")
        print(f"      Persona: {result.name}")
        print(f"      ID: {result.person_id}")
        print(f"      Score: {result.score:.4f}")
        print(f"      Distancia: {result.distance:.4f}")
        
        if result.person_id == "P001":
            print(f"\n   🎯 ¡CORRECTO! Identificó a la persona correcta")
        else:
            print(f"\n   ❌ ERROR: Identificó a la persona incorrecta")
    else:
        print(f"   ❌ No se encontró match")
        print(f"      Distancia: {result.distance:.4f}")
    
    print("\n✅ Test completado!")

if __name__ == "__main__":
    main()
