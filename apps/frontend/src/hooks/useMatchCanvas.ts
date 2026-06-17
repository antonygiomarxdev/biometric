import { useEffect, useRef } from "react";
import type { MinutiaPoint, MatchTraceEntry } from "@/lib/api";

/**
 * useMatchCanvas — dual-canvas + SVG line overlay (Phase 23, D-04/D-05/D-06/D-09).
 *
 * Renders two synchronized <canvas> elements (probe left, candidate right)
 * with minutia dots colored per match, plus a single <svg> overlay
 * spanning the container that draws one <line> per matched cylinder
 * pair. The line color comes from a 10-color cyclic palette
 * (`colorForIndex`); the line opacity is proportional to the
 * similarity (D-06).
 *
 * Coordinate system: internal pixel coords on each canvas with
 * `object-fit: contain`. The SVG line endpoints are computed in
 * display rect coordinates so the lines track the canvas position
 * and size on window resize.
 */
export interface UseMatchCanvasArgs {
  probeImageUrl: string;
  probeMinutiae: MinutiaPoint[];
  candidateImageUrl: string;
  candidateMinutiae: MinutiaPoint[];
  matchTrace: MatchTraceEntry[];
  containerRef: React.RefObject<HTMLDivElement | null>;
}

export interface UseMatchCanvasResult {
  probeCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  candidateCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  svgRef: React.RefObject<SVGSVGElement | null>;
}

// Locked 10-color cyclic palette (UI-SPEC §Color). Tailwind 500-series
// for high contrast on the dark canvas backdrop. Deterministic by
// index; never use Math.random() (RESEARCH §Pitfall 9).
export const PALETTE: readonly string[] = [
  "#ef4444", // 0  red-500
  "#22c55e", // 1  green-500
  "#3b82f6", // 2  blue-500
  "#eab308", // 3  yellow-500
  "#a855f7", // 4  purple-500
  "#ec4899", // 5  pink-500
  "#14b8a6", // 6  teal-500
  "#f97316", // 7  orange-500
  "#06b6d4", // 8  cyan-500
  "#84cc16", // 9  lime-500
] as const;

export function colorForIndex(i: number): string {
  return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length]!;
}

const MINUTIA_RADIUS = 3.5;
const MATCHED_RING_RADIUS = 5.0;
const SVG_NS = "http://www.w3.org/2000/svg";

function drawDot(
  ctx: CanvasRenderingContext2D,
  m: MinutiaPoint,
  color: string,
  ringRadius: number,
  innerRadius: number,
): void {
  // Outer ring (colored)
  ctx.beginPath();
  ctx.arc(m.x, m.y, ringRadius, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = 1;
  ctx.stroke();
  // Inner dot (white)
  ctx.beginPath();
  ctx.arc(m.x, m.y, innerRadius, 0, 2 * Math.PI);
  ctx.fillStyle = "white";
  ctx.fill();
}

function drawMinutiae(
  ctx: CanvasRenderingContext2D,
  minutiae: MinutiaPoint[],
  matched: Set<number>,
  pairColors: Map<number, string>,
): void {
  minutiae.forEach((m, idx) => {
    if (matched.has(idx)) {
      const pairColor = pairColors.get(idx) ?? PALETTE[0]!;
      drawDot(ctx, m, pairColor, MATCHED_RING_RADIUS, MINUTIA_RADIUS);
    } else {
      const typeColor = m.type === 1 ? "#22c55e" : "#ef4444";
      drawDot(ctx, m, typeColor, MINUTIA_RADIUS + 1, MINUTIA_RADIUS - 0.5);
    }
  });
}

function drawImageToCanvas(
  canvas: HTMLCanvasElement,
  imageUrl: string,
): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Failed to acquire 2d context"));
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => {
      // Empty 400x400 dark fallback so the canvas still renders
      canvas.width = 400;
      canvas.height = 400;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, 400, 400);
      }
      resolve({ width: 400, height: 400 });
    };
    img.src = imageUrl;
  });
}

export function useMatchCanvas(args: UseMatchCanvasArgs): UseMatchCanvasResult {
  const probeCanvasRef = useRef<HTMLCanvasElement>(null);
  const candidateCanvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Draw canvases whenever images or minutiae change.
  useEffect(() => {
    const probeCanvas = probeCanvasRef.current;
    const candidateCanvas = candidateCanvasRef.current;
    if (!probeCanvas || !candidateCanvas) return;

    const probeMatched = new Set<number>();
    const probePairColors = new Map<number, string>();
    args.matchTrace.forEach((entry, pairIndex) => {
      probeMatched.add(entry.probe_cylinder_index);
      probePairColors.set(entry.probe_cylinder_index, colorForIndex(pairIndex));
    });

    let cancelled = false;
    Promise.all([
      drawImageToCanvas(probeCanvas, args.probeImageUrl),
      drawImageToCanvas(candidateCanvas, args.candidateImageUrl),
    ]).then(() => {
      if (cancelled) return;
      const probeCtx = probeCanvas.getContext("2d");
      const candidateCtx = candidateCanvas.getContext("2d");
      if (!probeCtx || !candidateCtx) return;

      // Probe dots
      drawMinutiae(probeCtx, args.probeMinutiae, probeMatched, probePairColors);

      // Candidate dots: draw the entry positions directly (one dot per
      // match pair, since the wire format gives (x, y) of the
      // candidate minutia that matched, not an index into the
      // candidate minutiae array).
      args.matchTrace.forEach((entry, pairIndex) => {
        const color = colorForIndex(pairIndex);
        drawDot(
          candidateCtx,
          { x: entry.candidate_x, y: entry.candidate_y, angle: 0, type: 0 },
          color,
          MATCHED_RING_RADIUS,
          MINUTIA_RADIUS,
        );
      });
      // Unmatched candidate minutiae
      const matchedXs = new Set(args.matchTrace.map((e) => `${e.candidate_x},${e.candidate_y}`));
      args.candidateMinutiae.forEach((m) => {
        if (!matchedXs.has(`${m.x},${m.y}`)) {
          const typeColor = m.type === 1 ? "#22c55e" : "#ef4444";
          drawDot(candidateCtx, m, typeColor, MINUTIA_RADIUS + 1, MINUTIA_RADIUS - 0.5);
        }
      });
    });

    return () => {
      cancelled = true;
    };
  }, [
    args.probeImageUrl,
    args.probeMinutiae,
    args.candidateImageUrl,
    args.candidateMinutiae,
    args.matchTrace,
  ]);

  // Draw SVG lines whenever match trace, container size, or canvas
  // display rect changes.
  useEffect(() => {
    const svg = svgRef.current;
    const container = args.containerRef.current;
    const probeCanvas = probeCanvasRef.current;
    const candidateCanvas = candidateCanvasRef.current;
    if (!svg || !container || !probeCanvas || !candidateCanvas) return;

    const drawLines = () => {
      // Clear existing <line> children
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      const containerRect = container.getBoundingClientRect();
      const probeRect = probeCanvas.getBoundingClientRect();
      const candidateRect = candidateCanvas.getBoundingClientRect();

      // Internal pixel dims for ratio
      const probeScaleX = probeRect.width / Math.max(probeCanvas.width, 1);
      const probeScaleY = probeRect.height / Math.max(probeCanvas.height, 1);
      const candidateScaleX = candidateRect.width / Math.max(candidateCanvas.width, 1);
      const candidateScaleY = candidateRect.height / Math.max(candidateCanvas.height, 1);

      // The SVG spans the container, so all coords are relative to the
      // container's top-left.
      const probeOffsetX = probeRect.left - containerRect.left;
      const probeOffsetY = probeRect.top - containerRect.top;
      const candidateOffsetX = candidateRect.left - containerRect.left;
      const candidateOffsetY = candidateRect.top - containerRect.top;

      args.matchTrace.forEach((entry, pairIndex) => {
        const x1 = probeOffsetX + entry.probe_x * probeScaleX;
        const y1 = probeOffsetY + entry.probe_y * probeScaleY;
        const x2 = candidateOffsetX + entry.candidate_x * candidateScaleX;
        const y2 = candidateOffsetY + entry.candidate_y * candidateScaleY;

        const line = document.createElementNS(SVG_NS, "line");
        line.setAttribute("x1", x1.toString());
        line.setAttribute("y1", y1.toString());
        line.setAttribute("x2", x2.toString());
        line.setAttribute("y2", y2.toString());
        line.setAttribute("stroke", colorForIndex(pairIndex));
        line.setAttribute("stroke-width", "1.5");
        line.setAttribute("stroke-opacity", String(Math.max(0, Math.min(1, entry.similarity))));
        line.setAttribute("stroke-linecap", "round");
        line.setAttribute("data-pair-index", String(pairIndex));
        svg.appendChild(line);
      });
    };

    drawLines();
    const ro = new ResizeObserver(() => drawLines());
    ro.observe(container);
    ro.observe(probeCanvas);
    ro.observe(candidateCanvas);
    return () => {
      ro.disconnect();
    };
  }, [args.matchTrace, args.containerRef, args.probeMinutiae, args.candidateMinutiae]);

  return { probeCanvasRef, candidateCanvasRef, svgRef };
}
