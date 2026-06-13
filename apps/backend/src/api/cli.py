"""CLI for the fingerprint identification system."""

import sys
import logging
from pathlib import Path
import click

from src.services.fingerprint_service import fingerprint_service
from src.storage.database import db_manager
from src.storage.repository import repository
from src.core.metrics import metrics
from src.core.config import config
from src.core.compliance import setup_compliance_logging

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Apply PII scrubbing now that the root logger has its handler.
# Falls back to BaseStrategy (no scrubbing) if not configured,
# preserving all existing CLI logging behaviour.
setup_compliance_logging()

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Biometric Fingerprint System - CLI"""
    pass


@cli.command()
def init_db():
    """Initialize the database."""
    click.echo("Initializing database...")
    db_manager.create_tables()
    click.echo("OK Database initialized")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def extract(image_path):
    """Extract minutiae from a fingerprint image."""
    try:
        click.echo(f"Processing: {image_path}")
        fingerprint = fingerprint_service.process_image_from_path(image_path)
        click.echo(f"\nOK Minutiae extracted: {len(fingerprint.minutiae)}")

        stats = metrics.get_stats("process_image_full")
        if stats:
            click.echo(f"\nPerformance:")
            click.echo(f"  Avg: {stats['mean']:.2f}ms")
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--person-id", required=True, help="Person unique ID")
@click.option("--name", required=True, help="Full name")
@click.option("--document", required=True, help="Document number")
def register(image_path, person_id, name, document):
    """Register a new fingerprint."""
    try:
        click.echo(f"Processing: {image_path}")
        fingerprint = fingerprint_service.process_image_from_path(image_path)

        if not fingerprint.minutiae:
            click.echo("ERROR: No minutiae extracted", err=True)
            sys.exit(1)

        minutiae_data = [
            {
                "x": m.x, "y": m.y,
                "type": m.type.value if hasattr(m.type, 'value') else m.type,
                "angle": m.angle, "confidence": m.confidence
            }
            for m in fingerprint.minutiae
        ]

        record_id = repository.register(
            fp=fingerprint,
            person_id=person_id,
            name=name,
            doc=document,
            image_path=None,
            minutiae_data=minutiae_data
        )

        click.echo(f"\nOK Fingerprint registered")
        click.echo(f"  Record ID: {record_id}")
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def identify(image_path):
    """Identify a fingerprint."""
    try:
        click.echo(f"Processing: {image_path}")
        fingerprint = fingerprint_service.process_image_from_path(image_path)

        if not fingerprint.minutiae:
            click.echo("ERROR: No minutiae extracted", err=True)
            sys.exit(1)

        click.echo("\nSearching matches...")
        result = repository.identify(fingerprint, top_k=config.top_k_matches)

        if result.matched:
            click.echo(f"\nOK MATCH FOUND")
            click.echo(f"  Person: {result.name}")
            click.echo(f"  Score: {result.score:.4f}")
        else:
            click.echo(f"\nNo match found")
            click.echo(f"  Min distance: {result.distance:.4f}")
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show system status."""
    try:
        from src.storage.vector_index import vector_index
        click.echo("System Status")
        click.echo("=" * 50)
        click.echo(f"\nDatabase Records: {repository.count()}")
        click.echo(f"\nVector Index:")
        click.echo(f"  Vectors: {vector_index.size()}")
        click.echo(f"  Dimension: {config.vector_dimension}")
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@cli.command()
def show_metrics():
    """Show performance metrics."""
    try:
        if not metrics.metrics:
            click.echo("\nNo metrics available")
            return
        for operation in metrics.metrics.keys():
            stats = metrics.get_stats(operation)
            if stats:
                click.echo(f"\n{operation}:")
                click.echo(f"  Count: {stats['count']}")
                click.echo(f"  Avg: {stats['mean']:.2f}ms")
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


@cli.command()
def reset_metrics():
    """Reset accumulated metrics."""
    metrics.reset()
    click.echo("OK Metrics reset")


if __name__ == "__main__":
    cli()
