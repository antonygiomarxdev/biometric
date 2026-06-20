import { useEffect, useRef } from "react";

export function useCanvas(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  imgSrc: string | null,
  drawCallback?: (ctx: CanvasRenderingContext2D, img: HTMLImageElement) => void
) {
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!imgSrc) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      
      // Debug: Fill with a solid color
      ctx.fillStyle = "red";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.drawImage(img, 0, 0);

      if (drawCallback) {
        drawCallback(ctx, img);
      }
    };
    img.onerror = () => {
      console.error("Failed to load image:", imgSrc);
    };
    img.src = imgSrc;
  }, [imgSrc, canvasRef, drawCallback]);
}
