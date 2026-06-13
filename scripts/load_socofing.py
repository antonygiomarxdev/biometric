"""Script para cargar y probar con el dataset SOCOFing."""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple
import random

import click
import cv2
import numpy as np

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.fingerprint_service import fingerprint_service
from src.storage.repository import repository

from src.storage.database import db_manager
from src.core.metrics import metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_socofing_filename(filename: str) -> dict:
    """Extrae información del nombre de archivo de SOCOFing.
    
    Formato: {person_id}__{gender}_{hand}_{finger}.BMP
    Ejemplo: 1__M_Left_index_finger.BMP
    
    Returns:
        dict con person_id, gender, hand, finger
    """
    parts = filename.replace(".BMP", "").split("__")
    if len(parts) != 2:
        return None
    
    person_id = parts[0]
    info_parts = parts[1].split("_", 2)
    
    if len(info_parts) < 3:
        return None
    
    return {
        "person_id": f"SOC_{person_id.zfill(4)}",
        "gender": info_parts[0],
        "hand": info_parts[1],
        "finger": info_parts[2].replace("_", " "),
        "document": f"DOC_{person_id.zfill(8)}",
        "name": f"Person {person_id}",
        "original_filename": filename
    }


def load_socofing_dataset(
    dataset_path: str,
    subset: str = "Real",
    limit: int = None
) -> List[Tuple[str, dict]]:
    """Carga las rutas de imágenes del dataset SOCOFing.
    
    Args:
        dataset_path: Ruta al dataset SOCOFing
        subset: 'Real', 'Altered-Easy', 'Altered-Medium', 'Altered-Hard'
        limit: Número máximo de imágenes a cargar
        
    Returns:
        Lista de tuplas (ruta_imagen, metadata)
    """
    subset_path = Path(dataset_path) / subset
    
    if not subset_path.exists():
        raise ValueError(f"No se encuentra el subset: {subset_path}")
    
    images = []
    for img_file in subset_path.glob("*.BMP"):
        metadata = parse_socofing_filename(img_file.name)
        if metadata:
            images.append((str(img_file), metadata))
    
    if limit:
        images = images[:limit]
    
    logger.info(f"Cargadas {len(images)} imágenes del subset '{subset}'")
    return images


@click.group()
def cli():
    """Script para trabajar con el dataset SOCOFing."""
    pass


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--subset", default="Real", help="Subset a usar (Real, Altered-Easy, etc)")
@click.option("--limit", type=int, help="Limitar número de imágenes")
@click.option("--skip-errors", is_flag=True, help="Continuar si hay errores")
def register(dataset_path, subset, limit, skip_errors):
    """Registra huellas del dataset SOCOFing en el sistema.
    
    Ejemplo:
        python scripts/load_socofing.py register data/SOCOFing --limit 100
    """
    click.echo("=" * 70)
    click.echo("📁 REGISTRO DE DATASET SOCOFING")
    click.echo("=" * 70)
    
    # Inicializar DB
    click.echo("\n🔧 Inicializando base de datos...")
    db_manager.create_tables()
    
    # Cargar dataset
    click.echo(f"\n📂 Cargando dataset desde: {dataset_path}")
    images = load_socofing_dataset(dataset_path, subset, limit)
    click.echo(f"   Total de imágenes: {len(images)}")
    
    # Registrar
    click.echo(f"\n🚀 Iniciando registro...")
    start_time = time.time()
    
    registered = 0
    errors = 0
    
    with click.progressbar(images, label="Registrando") as bar:
        for img_path, metadata in bar:
            try:
                # Leer imagen
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"No se pudo leer la imagen")
                
                # Procesar
                fingerprint = fingerprint_service.process_image(
                    img,
                    fingerprint_id=metadata["original_filename"]
                )
                
                # Verificar que tiene minutiae
                if not fingerprint.minutiae:
                    if not skip_errors:
                        raise ValueError("No se extrajeron minutiae")
                    logger.warning(f"Sin minutiae: {metadata['original_filename']}")
                    errors += 1
                    continue
                
                # Registrar
                repository.register(fp=
                    fingerprint=fingerprint,
                    person_id=metadata["person_id"],
                    name=f"{metadata['name']} - {metadata['hand']} {metadata['finger']}",
                    document=metadata["document"]
                )
                
                registered += 1
                
            except Exception as e:
                errors += 1
                if skip_errors:
                    logger.error(f"Error en {metadata['original_filename']}: {e}")
                else:
                    raise
    
    # Estadísticas
    duration = time.time() - start_time
    
    click.echo("\n" + "=" * 70)
    click.echo("📊 ESTADÍSTICAS DE REGISTRO")
    click.echo("=" * 70)
    click.echo(f"✅ Registradas exitosamente: {registered}")
    click.echo(f"❌ Errores: {errors}")
    click.echo(f"⏱️  Tiempo total: {duration:.2f}s")
    click.echo(f"⚡ Promedio por huella: {(duration/registered):.2f}s")
    
    # Métricas
    if metrics.metrics:
        click.echo("\n📈 MÉTRICAS DE PERFORMANCE")
        click.echo("-" * 70)
        
        for operation in ["process_image_full", "repository_register"]:
            stats = metrics.get_stats(operation)
            if stats:
                click.echo(f"\n{operation}:")
                click.echo(f"  - Promedio: {stats['mean']:.2f}ms")
                click.echo(f"  - P50: {stats['p50']:.2f}ms")
                click.echo(f"  - P95: {stats['p95']:.2f}ms")
                click.echo(f"  - Min/Max: {stats['min']:.2f}ms / {stats['max']:.2f}ms")


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--subset", default="Real", help="Subset para identificar")
@click.option("--num-queries", type=int, default=10, help="Número de huellas a identificar")
@click.option("--different-person", is_flag=True, help="Probar con personas no registradas")
def identify(dataset_path, subset, num_queries, different_person):
    """Identifica huellas del dataset y calcula accuracy.
    
    Ejemplo:
        python scripts/load_socofing.py identify data/SOCOFing --num-queries 50
    """
    click.echo("=" * 70)
    click.echo("🔍 IDENTIFICACIÓN DE HUELLAS")
    click.echo("=" * 70)
    
    # Cargar imágenes
    images = load_socofing_dataset(dataset_path, subset)
    
    if len(images) == 0:
        click.echo("❌ No hay imágenes en el dataset")
        return
    
    # Seleccionar imágenes al azar
    test_images = random.sample(images, min(num_queries, len(images)))
    
    click.echo(f"\n🎯 Identificando {len(test_images)} huellas...")
    
    results = {
        "total": 0,
        "matched": 0,
        "correct": 0,
        "incorrect": 0,
        "not_found": 0,
        "distances": [],
        "scores": []
    }
    
    start_time = time.time()
    
    with click.progressbar(test_images, label="Identificando") as bar:
        for img_path, metadata in bar:
            try:
                # Leer y procesar imagen
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                fingerprint = fingerprint_service.process_image(img)
                
                if not fingerprint.minutiae:
                    continue
                
                # Identificar
                result = repository.identify(fingerprint)
                
                results["total"] += 1
                
                if result.matched:
                    results["matched"] += 1
                    results["distances"].append(result.distance)
                    results["scores"].append(result.score)
                    
                    # Verificar si es correcta
                    if result.person_id == metadata["person_id"]:
                        results["correct"] += 1
                    else:
                        results["incorrect"] += 1
                        logger.warning(
                            f"Match incorrecto: {metadata['person_id']} → {result.person_id}"
                        )
                else:
                    results["not_found"] += 1
                
            except Exception as e:
                logger.error(f"Error identificando {metadata['original_filename']}: {e}")
    
    duration = time.time() - start_time
    
    # Estadísticas
    click.echo("\n" + "=" * 70)
    click.echo("📊 RESULTADOS DE IDENTIFICACIÓN")
    click.echo("=" * 70)
    click.echo(f"Total procesadas: {results['total']}")
    click.echo(f"✅ Coincidencias encontradas: {results['matched']}")
    click.echo(f"   - Correctas: {results['correct']}")
    click.echo(f"   - Incorrectas: {results['incorrect']}")
    click.echo(f"❌ No encontradas: {results['not_found']}")
    
    if results["total"] > 0:
        accuracy = (results["correct"] / results["total"]) * 100
        click.echo(f"\n🎯 ACCURACY: {accuracy:.2f}%")
    
    if results["distances"]:
        click.echo(f"\n📏 Distancias:")
        click.echo(f"   - Promedio: {np.mean(results['distances']):.4f}")
        click.echo(f"   - Mínima: {np.min(results['distances']):.4f}")
        click.echo(f"   - Máxima: {np.max(results['distances']):.4f}")
    
    if results["scores"]:
        click.echo(f"\n⭐ Scores:")
        click.echo(f"   - Promedio: {np.mean(results['scores']):.4f}")
        click.echo(f"   - Mínimo: {np.min(results['scores']):.4f}")
        click.echo(f"   - Máximo: {np.max(results['scores']):.4f}")
    
    click.echo(f"\n⏱️  Tiempo total: {duration:.2f}s")
    click.echo(f"⚡ Promedio por búsqueda: {(duration/results['total']):.3f}s")


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def stats(dataset_path):
    """Muestra estadísticas del dataset SOCOFing."""
    click.echo("=" * 70)
    click.echo("📊 ESTADÍSTICAS DEL DATASET SOCOFING")
    click.echo("=" * 70)
    
    for subset in ["Real", "Altered-Easy", "Altered-Medium", "Altered-Hard"]:
        try:
            images = load_socofing_dataset(dataset_path, subset)
            click.echo(f"\n{subset}:")
            click.echo(f"  - Total de imágenes: {len(images)}")
            
            if images:
                # Contar personas únicas
                persons = set(m["person_id"] for _, m in images)
                click.echo(f"  - Personas únicas: {len(persons)}")
                
                # Contar por género
                genders = {}
                for _, m in images:
                    genders[m["gender"]] = genders.get(m["gender"], 0) + 1
                click.echo(f"  - Por género: {dict(genders)}")
                
        except ValueError:
            click.echo(f"\n{subset}: No encontrado")


if __name__ == "__main__":
    cli()
