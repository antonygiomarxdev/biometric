"""CLI para el sistema de huellas dactilares."""

import sys
import logging
from pathlib import Path
import click

from src.services.fingerprint_service import fingerprint_service
from src.services.comparison_service import comparison_service
from src.storage.database import db_manager
from src.storage.repository import repository
from src.core.metrics import metrics
from src.core.config import config

# Configurar logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Sistema Biométrico de Huellas Dactilares - CLI"""
    pass


@cli.command()
def init_db():
    """Inicializa la base de datos creando las tablas."""
    click.echo("Inicializando base de datos...")
    db_manager.create_tables()
    click.echo("✓ Base de datos inicializada correctamente")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def extract(image_path):
    """Extrae minutiae de una imagen de huella.
    
    Args:
        image_path: Ruta a la imagen
    """
    try:
        click.echo(f"Procesando: {image_path}")
        
        fingerprint = fingerprint_service.process_image_from_path(image_path)
        
        click.echo(f"\n✓ Minutiae extraídas: {len(fingerprint.minutiae)}")
        click.echo(f"  - Terminaciones: {sum(1 for m in fingerprint.minutiae if m.type == 'termination')}")
        click.echo(f"  - Bifurcaciones: {sum(1 for m in fingerprint.minutiae if m.type == 'bifurcation')}")
        
        # Mostrar métricas
        stats = metrics.get_stats("process_image_full")
        if stats:
            click.echo(f"\nPerformance:")
            click.echo(f"  - Tiempo promedio: {stats['mean']:.2f}ms")
            click.echo(f"  - P50: {stats['p50']:.2f}ms")
            click.echo(f"  - P95: {stats['p95']:.2f}ms")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--person-id", required=True, help="ID único de la persona")
@click.option("--name", required=True, help="Nombre completo")
@click.option("--document", required=True, help="Número de documento")
def register(image_path, person_id, name, document):
    """Registra una nueva huella en el sistema.
    
    Args:
        image_path: Ruta a la imagen de la huella
    """
    try:
        click.echo(f"Procesando imagen: {image_path}")
        
        # Procesar imagen
        fingerprint = fingerprint_service.process_image_from_path(image_path)
        
        if not fingerprint.minutiae:
            click.echo("✗ No se pudieron extraer minutiae", err=True)
            sys.exit(1)
        
        click.echo(f"Minutiae extraídas: {len(fingerprint.minutiae)}")
        
        # Registrar
        record_id = comparison_service.register_fingerprint(
            fingerprint=fingerprint,
            person_id=person_id,
            name=name,
            document=document
        )
        
        click.echo(f"\n✓ Huella registrada exitosamente")
        click.echo(f"  - Record ID: {record_id}")
        click.echo(f"  - Persona: {name} ({person_id})")
        click.echo(f"  - Documento: {document}")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def identify(image_path):
    """Identifica una huella buscando coincidencias.
    
    Args:
        image_path: Ruta a la imagen de la huella
    """
    try:
        click.echo(f"Procesando imagen: {image_path}")
        
        # Procesar imagen
        fingerprint = fingerprint_service.process_image_from_path(image_path)
        
        if not fingerprint.minutiae:
            click.echo("✗ No se pudieron extraer minutiae", err=True)
            sys.exit(1)
        
        click.echo(f"Minutiae extraídas: {len(fingerprint.minutiae)}")
        click.echo("\nBuscando coincidencias...")
        
        # Identificar
        result = comparison_service.identify(fingerprint)
        
        if result.matched:
            click.echo(f"\n✓ COINCIDENCIA ENCONTRADA")
            click.echo(f"  - Persona: {result.name}")
            click.echo(f"  - ID: {result.person_id}")
            click.echo(f"  - Documento: {result.document}")
            click.echo(f"  - Score: {result.score:.4f}")
            click.echo(f"  - Distancia: {result.distance:.4f}")
        else:
            click.echo(f"\n✗ No se encontraron coincidencias")
            click.echo(f"  - Distancia mínima: {result.distance:.4f}")
            click.echo(f"  - Umbral: {config.match_threshold}")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Muestra el estado del sistema."""
    try:
        from src.storage.vector_index import vector_index
        
        click.echo("Estado del Sistema")
        click.echo("=" * 50)
        click.echo(f"\nBase de Datos:")
        click.echo(f"  - Registros: {repository.count()}")
        click.echo(f"\nÍndice Vectorial:")
        click.echo(f"  - Backend: pgvector")
        click.echo(f"  - Vectores: {vector_index.size()}")
        click.echo(f"  - Dimensión: {config.vector_dimension}")
        click.echo(f"\nConfiguración:")
        click.echo(f"  - Umbral de match: {config.match_threshold}")
        click.echo(f"  - Top K matches: {config.top_k_matches}")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def show_metrics():
    """Muestra las métricas de performance."""
    try:
        click.echo("Métricas de Performance")
        click.echo("=" * 50)
        
        if not metrics.metrics:
            click.echo("\nNo hay métricas disponibles")
            return
        
        for operation in metrics.metrics.keys():
            stats = metrics.get_stats(operation)
            if stats:
                click.echo(f"\n{operation}:")
                click.echo(f"  - Ejecuciones: {stats['count']}")
                click.echo(f"  - Promedio: {stats['mean']:.2f}ms")
                click.echo(f"  - P50: {stats['p50']:.2f}ms")
                click.echo(f"  - P95: {stats['p95']:.2f}ms")
                click.echo(f"  - P99: {stats['p99']:.2f}ms")
                click.echo(f"  - Min/Max: {stats['min']:.2f}ms / {stats['max']:.2f}ms")
    
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def reset_metrics():
    """Resetea las métricas acumuladas."""
    metrics.reset()
    click.echo("✓ Métricas reseteadas")


if __name__ == "__main__":
    cli()
