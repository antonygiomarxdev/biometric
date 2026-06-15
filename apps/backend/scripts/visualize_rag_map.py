"""
Generate visual map of the RAG Fingerprint Triangulation.
This script extracts minutiae, finds the core, computes Delaunay triangles,
and plots them with a heat map based on their weight (proximity to core).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from src.core.interfaces import PipelineContext
from src.services.fingerprint_service import FingerprintService
from src.processing.vectorizer import RagTripletVectorizer
from src.domain.forensic_rules import SearchValidationStrategy


def main():
    print("Loading image...")
    img_path = Path("static/SOCOFing/Real/100__M_Left_index_finger.BMP")
    if not img_path.exists():
        print(f"Error: {img_path} not found")
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    print("Running pipeline...")
    # Run full pipeline to get candidates and core
    svc = FingerprintService(validation_strategy=SearchValidationStrategy())
    ctx = PipelineContext(raw_image=img, fingerprint_id="viz")
    
    # Run step by step to populate context
    for step in svc.steps:
        step.process(ctx)
        # We don't need normalizer for this visualization
        if type(step).__name__ == "NormalizerStep":
            break

    print(f"Detected {len(ctx.candidates)} minutiae. Core: {ctx.core}")

    print("Generating RAG Triangulation Map...")
    vec = RagTripletVectorizer(sigma=80.0)
    points, types = vec._extract_points_and_types(ctx.candidates)
    triangles = vec._delaunay_triangulate(points)
    
    centroids = np.array([points[list(tri)].mean(axis=0) for tri in triangles])
    weights = vec._compute_weights(centroids, ctx.core)

    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#0d1117') # GitHub dark mode background
    
    # -------------------------------------------------------------
    # Plot 1: Minutiae & Core
    # -------------------------------------------------------------
    ax1.set_facecolor('#0d1117')
    ax1.imshow(ctx.enhanced_image, cmap='gray', alpha=0.8)
    
    # Plot Minutiae
    terms = points[types == 0]
    bifs = points[types == 1]
    ax1.scatter(terms[:,0], terms[:,1], c='#00ff00', s=30, label='Terminations')
    ax1.scatter(bifs[:,0], bifs[:,1], c='#ff00ff', s=30, marker='s', label='Bifurcations')
    
    # Plot Core
    if ctx.core:
        ax1.scatter(ctx.core[0], ctx.core[1], c='#ff0000', marker='*', s=600, edgecolors='white', linewidths=1.5, label='Core (Singularity)')
    
    ax1.set_title("1. Detección de Minucias y Núcleo", color='white', fontsize=18, pad=20)
    ax1.legend(loc='upper right', facecolor='#161b22', edgecolor='white', labelcolor='white')
    ax1.axis('off')

    # -------------------------------------------------------------
    # Plot 2: RAG Map (Delaunay Triplets + Heatmap)
    # -------------------------------------------------------------
    ax2.set_facecolor('#0d1117')
    ax2.imshow(ctx.enhanced_image, cmap='gray', alpha=0.3)
    
    polys = [points[list(tri)] for tri in triangles]
    
    # Colormap from blue (weight=0) to red (weight=1)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    
    collection = PolyCollection(polys, array=np.array(weights), cmap=cmap, norm=norm, 
                                edgecolors='white', linewidths=0.5, alpha=0.6)
    ax2.add_collection(collection)
    
    if ctx.core:
        ax2.scatter(ctx.core[0], ctx.core[1], c='#ffffff', marker='*', s=400, edgecolors='black', label='Core')
        
    ax2.set_title("2. Chunking (Delaunay) & Pesos Forenses", color='white', fontsize=18, pad=20)
    ax2.axis('off')
    
    cbar = fig.colorbar(collection, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Peso Forense (Importancia para RAG)', color='white', size=14)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    out_path = Path("../../docs/assets/rag_map.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
