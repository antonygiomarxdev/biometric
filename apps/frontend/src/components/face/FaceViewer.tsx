import { Camera, RefreshCw } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export function FaceViewer() {
  return (
    <div className="col-span-6 flex flex-col h-full">
      <Card className="flex flex-col h-full shadow-md border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden">
        <CardContent className="flex-1 p-0 relative bg-black flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <div className="w-32 h-32 mx-auto mb-6 rounded-full bg-white/5 flex items-center justify-center border-2 border-dashed border-white/20">
              <Camera className="w-12 h-12 opacity-50" />
            </div>
            <h3 className="text-xl font-medium text-white mb-2">Reconocimiento Facial</h3>
            <p className="text-sm opacity-60 max-w-xs mx-auto mb-6">
              Módulo preparado para integración con cámara web y detección de vivacidad.
            </p>
            <Button variant="secondary" className="gap-2">
              <RefreshCw className="w-4 h-4" />
              Activar Cámara
            </Button>
          </div>
          
          {/* Overlay de guías faciales simuladas */}
          <div className="absolute inset-0 pointer-events-none opacity-20">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-80 border-2 border-primary rounded-[50%]" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
