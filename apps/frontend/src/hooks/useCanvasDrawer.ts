import { useRef, useEffect, useState, useCallback } from "react";
import type { MinutiaPoint } from "@/lib/api";

export type EditingMode = "view" | "add" | "delete" | "move";

export interface EditingState {
  mode: EditingMode;
  selectedIndex: number | null;
  minutiae: MinutiaPoint[];
}

export interface CanvasDrawerResult {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  editingState: EditingState;
  setMode: (mode: EditingMode) => void;
  setMinutiae: (minutiae: MinutiaPoint[]) => void;
  handleCanvasClick: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  handleMouseMove: (e: React.MouseEvent<HTMLCanvasElement>) => void;
  handleSave: () => MinutiaPoint[];
}

const HIT_RADIUS = 15;
const MINUTIA_RADIUS = 3;

const NEAREST_NOT_FOUND = -1;

function getCanvasCoordinates(
  e: React.MouseEvent<HTMLCanvasElement>,
  canvas: HTMLCanvasElement
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function findNearestMinutiaIndex(
  x: number,
  y: number,
  minutiae: MinutiaPoint[]
): number {
  let minDist = Infinity;
  let minIdx = NEAREST_NOT_FOUND;

  for (let i = 0; i < minutiae.length; i++) {
    const m = minutiae[i];
    const dx = m.x - x;
    const dy = m.y - y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < minDist) {
      minDist = dist;
      minIdx = i;
    }
  }

  return minDist <= HIT_RADIUS ? minIdx : NEAREST_NOT_FOUND;
}

export const useCanvasDrawer = (
  selectedFingerId: string | null,
  previewUrl: string | undefined,
  extractData:
    | { minutiae: MinutiaPoint[]; processed_image?: string | null }
    | undefined
): CanvasDrawerResult => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationTimeouts = useRef<ReturnType<typeof setTimeout>[]>([]);
  const imageRef = useRef<string | undefined>(undefined);

  const [editingState, setEditingState] = useState<EditingState>({
    mode: "view",
    selectedIndex: null,
    minutiae: [],
  });

  // Track a drag offset for move mode: how far from the minutia center the
  // user initially clicked, so the drag feels anchored to the cursor.
  const dragOffset = useRef<{ dx: number; dy: number }>({ dx: 0, dy: 0 });

  const clearAnimations = useCallback(() => {
    animationTimeouts.current.forEach(clearTimeout);
    animationTimeouts.current = [];
  }, []);

  const drawCanvas = useCallback(
    (
      minutiaeList: MinutiaPoint[],
      imageSource: string | undefined,
      animate: boolean,
      highlightIndex: number | null
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      clearAnimations();

      const img = new Image();
      img.src = imageSource || "";

      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        if (animate) {
          minutiaeList.forEach((m, index) => {
            const timeoutId = setTimeout(() => {
              drawMinutiaDot(ctx, m, index === highlightIndex);
            }, index * 5);
            animationTimeouts.current.push(timeoutId);
          });
        } else {
          minutiaeList.forEach((m, index) => {
            drawMinutiaDot(ctx, m, index === highlightIndex);
          });
        }
      };

      // Handle error loading image — draw minutiae on empty canvas
      img.onerror = () => {
        canvas.width = 400;
        canvas.height = 400;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        if (!animate) {
          minutiaeList.forEach((m, index) => {
            drawMinutiaDot(ctx, m, index === highlightIndex);
          });
        }
      };
    },
    [clearAnimations]
  );

  const redraw = useCallback(() => {
    const imageSource = imageRef.current;
    if (!imageSource) return;
    drawCanvas(
      editingState.minutiae,
      imageSource,
      false,
      editingState.selectedIndex
    );
  }, [drawCanvas, editingState.minutiae, editingState.selectedIndex]);

  // View-mode effect: re-draw when props change (with animation)
  useEffect(() => {
    if (editingState.mode !== "view") return;

    const imageBase64 = extractData?.processed_image ?? null;
    const imageSource = imageBase64
      ? `data:image/png;base64,${imageBase64}`
      : previewUrl;

    imageRef.current = imageSource;

    if (extractData?.minutiae) {
      setEditingState((prev) => ({
        ...prev,
        minutiae: extractData.minutiae!,
      }));
      drawCanvas(extractData.minutiae, imageSource, true, null);
    } else if (imageSource) {
      drawCanvas([], imageSource, false, null);
    }

    return () => {
      clearAnimations();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFingerId, previewUrl, extractData]);

  const setMode = useCallback((mode: EditingMode) => {
    setEditingState((prev) => ({ ...prev, mode, selectedIndex: null }));
  }, []);

  const setMinutiae = useCallback((minutiae: MinutiaPoint[]) => {
    setEditingState((prev) => ({ ...prev, minutiae }));
  }, []);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const { x, y } = getCanvasCoordinates(e, canvas);
      const isShiftKey = e.shiftKey;

      setEditingState((prev) => {
        let newMinutiae = [...prev.minutiae];
        let newSelected: number | null = prev.selectedIndex;

        switch (prev.mode) {
          case "add": {
            const newPoint: MinutiaPoint = {
              x,
              y,
              type: isShiftKey ? 1 : 0,
              angle: 0,
            };
            newMinutiae.push(newPoint);
            newSelected = null;
            break;
          }

          case "delete": {
            const idx = findNearestMinutiaIndex(x, y, newMinutiae);
            if (idx !== NEAREST_NOT_FOUND) {
              newMinutiae.splice(idx, 1);
              newSelected = null;
            }
            break;
          }

          case "move": {
            const idx = findNearestMinutiaIndex(x, y, newMinutiae);
            if (idx !== NEAREST_NOT_FOUND) {
              // Select, but actual movement happens in handleMouseMove
              newSelected = idx;
              // Record offset so drag feels anchored
              const m = newMinutiae[idx];
              dragOffset.current = { dx: x - m.x, dy: y - m.y };
            } else {
              newSelected = null;
            }
            break;
          }

          default:
            break;
        }

        // Draw immediately after mutation (no animation)
        const imageSource = imageRef.current;
        if (imageSource) {
          drawCanvas(newMinutiae, imageSource, false, newSelected);
        }

        return { ...prev, minutiae: newMinutiae, selectedIndex: newSelected };
      });
    },
    [drawCanvas]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      setEditingState((prev) => {
        if (prev.mode !== "move" || prev.selectedIndex === null) return prev;

        const { x, y } = getCanvasCoordinates(e, canvas);
        const newMinutiae = [...prev.minutiae];
        const selectedMinutia = { ...newMinutiae[prev.selectedIndex] };

        // Apply drag offset so the minutia stays under cursor
        selectedMinutia.x = x - dragOffset.current.dx;
        selectedMinutia.y = y - dragOffset.current.dy;
        newMinutiae[prev.selectedIndex] = selectedMinutia;

        const imageSource = imageRef.current;
        if (imageSource) {
          drawCanvas(newMinutiae, imageSource, false, prev.selectedIndex);
        }

        return { ...prev, minutiae: newMinutiae };
      });
    },
    [drawCanvas]
  );

  const handleSave = useCallback((): MinutiaPoint[] => {
    return editingState.minutiae;
  }, [editingState.minutiae]);

  return {
    canvasRef,
    editingState,
    setMode,
    setMinutiae,
    handleCanvasClick,
    handleMouseMove,
    handleSave,
  };
};

function drawMinutiaDot(
  ctx: CanvasRenderingContext2D,
  m: MinutiaPoint,
  isHighlighted: boolean
): void {
  ctx.beginPath();
  ctx.arc(m.x, m.y, MINUTIA_RADIUS, 0, 2 * Math.PI);
  ctx.fillStyle = m.type === 0 ? "#ef4444" : "#22c55e";
  ctx.fill();

  // Highlighted minutia gets a yellow ring
  if (isHighlighted) {
    ctx.strokeStyle = "#eab308";
    ctx.lineWidth = 2.5;
  } else {
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;
  }
  ctx.stroke();
}
