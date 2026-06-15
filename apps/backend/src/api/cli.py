"""CLI for the fingerprint identification system."""

import subprocess
import sys
from pathlib import Path

import click

from src.core.compliance import setup_compliance_logging
from src.core.config import config
from src.core.metrics import metrics
from src.services.fingerprint_service import fingerprint_service

import logging

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
    """Apply database migrations via Alembic."""
    click.echo("Applying database migrations...")
    backend_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=str(backend_root / "src"),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"ERROR: {result.stderr}", err=True)
        sys.exit(result.returncode)
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
    """Register a fingerprint using the new services."""
    click.echo("This command has been migrated to the new service layer.")
    click.echo("Use POST /api/v1/known-fingerprints/ instead.")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
def identify(image_path):
    """Identify a fingerprint using the new services."""
    click.echo("This command has been migrated to the new service layer.")
    click.echo("Use POST /api/v1/matching/search instead.")


@cli.command()
def status():
    """Show system status."""
    click.echo("System Status")
    click.echo("=" * 50)
    click.echo(f"\nThis feature requires the new API service layer.")
    click.echo(f"Use an HTTP client to query the API for status.")


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
