import { useEffect, useRef } from "react";

export function useCanvas(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  imgSrc: string | null,
  drawCallback?: (ctx: CanvasRenderingContext2D, img: HTMLImageElement) => void
) {
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    console.log("useCanvas useEffect called with imgSrc:", imgSrc);
    const canvas = canvasRef.current;
    if (!canvas) {
      console.log("useCanvas: canvas is null");
      return;
    }
    if (!imgSrc) {
      console.log("useCanvas: imgSrc is null, clearing canvas");
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    console.log("useCanvas: creating new Image");
    const img = new Image();
    img.onload = () => {
      console.log("useCanvas: img.onload called");
      imgRef.current = img;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.fillStyle = "red";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.drawImage(img, 0, 0);

      if (drawCallback) {
        drawCallback(ctx, img);
      }
    };
    img.onerror = () => {
      console.error("useCanvas: Failed to load image:", imgSrc);
    };
    console.log("useCanvas: setting img.src to:", imgSrc);
    img.src = imgSrc;
  }, [imgSrc, canvasRef, drawCallback]);
}
