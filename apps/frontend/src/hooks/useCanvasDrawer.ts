import { useRef, useEffect } from "react";
import type { MinutiaPoint } from "../client";

export const useCanvasDrawer = (
  selectedFingerId: string | null,
  previewUrl: string | undefined,
  extractData:
    | { minutiae: MinutiaPoint[]; processed_image?: string | null }
    | undefined
) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationTimeouts = useRef<ReturnType<typeof setTimeout>[]>([]);

  const drawMinutiae = (
    minutiaeList: MinutiaPoint[],
    imageBase64?: string | null,
    basePreviewUrl?: string
  ) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear previous timeouts
    animationTimeouts.current.forEach(clearTimeout);
    animationTimeouts.current = [];

    const img = new Image();
    // Use processed image if available, otherwise original preview
    img.src = imageBase64
      ? `data:image/png;base64,${imageBase64}`
      : basePreviewUrl || "";

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Animate minutiae
      minutiaeList.forEach((m, index) => {
        const timeoutId = setTimeout(() => {
          ctx.beginPath();
          ctx.arc(m.x, m.y, 3, 0, 2 * Math.PI);
          ctx.fillStyle = m.type === 0 ? "#ef4444" : "#22c55e"; // Red: termination, Green: bifurcation
          ctx.fill();
          // Add white border for visibility
          ctx.strokeStyle = "white";
          ctx.lineWidth = 1;
          ctx.stroke();
        }, index * 5); // 5ms delay per point
        animationTimeouts.current.push(timeoutId);
      });
    };
  };

  useEffect(() => {
    if (extractData?.minutiae) {
      drawMinutiae(
        extractData.minutiae,
        extractData.processed_image,
        previewUrl
      );
    } else if (previewUrl) {
      // Just draw the image if no minutiae data
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        const img = new Image();
        img.src = previewUrl;
        img.onload = () => {
          canvas.width = img.width;
          canvas.height = img.height;
          ctx?.drawImage(img, 0, 0);
        };
      }
    }

    return () => {
      animationTimeouts.current.forEach(clearTimeout);
    };
  }, [selectedFingerId, previewUrl, extractData]);

  return canvasRef;
};
