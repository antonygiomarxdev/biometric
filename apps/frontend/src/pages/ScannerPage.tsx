import { useState, useRef } from "react";
import {
  Upload,
  Fingerprint,
  Search,
  UserPlus,
  CheckCircle,
  XCircle,
  Loader2,
  ArrowLeft,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DefaultService, OpenAPI } from "../client";
import type { MinutiaPoint, IdentifyResponse } from "../client";
import { useToast } from "@/components/ui/toast";
import { logger } from "@/lib/logger";
import { ApiError } from "../client/core/ApiError";

// Configure API Base URL
OpenAPI.BASE = "http://localhost:8000";

export default function ScannerPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<"scan" | "register">("scan");
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [, setMinutiae] = useState<MinutiaPoint[]>([]);
  const [result, setResult] = useState<IdentifyResponse | null>(null);

  // Registration form
  const [regName, setRegName] = useState("");
  const [regId, setRegId] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { addToast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      logger.info("Archivo seleccionado", {
        name: file.name,
        size: file.size,
        type: file.type,
      });

      // Validar tipo de archivo
      const validTypes = ["image/bmp", "image/png", "image/jpeg", "image/jpg"];
      if (!validTypes.includes(file.type)) {
        logger.warn("Tipo de archivo no válido", { type: file.type });
        addToast({
          type: "error",
          title: "Tipo de archivo inválido",
          description: "Por favor, selecciona una imagen BMP, PNG o JPEG",
        });
        return;
      }

      // Validar tamaño (máx 10MB)
      if (file.size > 10 * 1024 * 1024) {
        logger.warn("Archivo demasiado grande", { size: file.size });
        addToast({
          type: "error",
          title: "Archivo demasiado grande",
          description: "El archivo no debe exceder 10MB",
        });
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        logger.debug("Imagen cargada correctamente");
        setImagePreview(e.target?.result as string);
        setMinutiae([]);
        setResult(null);
        addToast({
          type: "success",
          title: "Imagen cargada",
          description: `Archivo ${file.name} listo para procesar`,
          duration: 3000,
        });
      };
      reader.onerror = () => {
        logger.error("Error leyendo archivo", reader.error);
        addToast({
          type: "error",
          title: "Error al leer archivo",
          description: "No se pudo cargar la imagen. Intenta nuevamente.",
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const drawMinutiae = (minutiaeList: MinutiaPoint[]) => {
    const canvas = canvasRef.current;
    if (!canvas || !imagePreview) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = imagePreview;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      minutiaeList.forEach((m) => {
        ctx.beginPath();
        ctx.arc(m.x, m.y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = m.type === 0 ? "#ef4444" : "#22c55e"; // Red for termination, Green for bifurcation
        ctx.fill();
      });
    };
  };

  const processImage = async (action: "identify" | "register") => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      logger.warn("Intento de procesar sin archivo seleccionado");
      addToast({
        type: "warning",
        title: "No hay archivo seleccionado",
        description: "Por favor, selecciona una imagen primero",
      });
      return;
    }

    logger.info(`Iniciando procesamiento: ${action}`, {
      fileName: file.name,
      fileSize: file.size,
    });
    setLoading(true);
    setResult(null);

    try {
      // 1. First extract minutiae for visualization
      logger.info("Extrayendo minutiae de la imagen...");
      addToast({
        type: "info",
        title: "Procesando imagen...",
        description: "Extrayendo características biométricas",
        duration: 2000,
      });

      const extractRes = await DefaultService.extractMinutiaeExtractPost({
        file,
      });

      logger.info("Minutiae extraídas", {
        count: extractRes.minutiae?.length || 0,
        terminations: extractRes.terminations || 0,
        bifurcations: extractRes.bifurcations || 0,
      });

      if (extractRes.minutiae && extractRes.minutiae.length > 0) {
        setMinutiae(extractRes.minutiae);
        drawMinutiae(extractRes.minutiae);
        logger.debug("Minutiae visualizadas en canvas");
      } else {
        logger.warn("No se encontraron minutiae en la imagen");
        addToast({
          type: "warning",
          title: "No se encontraron minutiae",
          description:
            "La imagen podría no ser válida o estar demasiado borrosa",
        });
      }

      if (action === "identify") {
        logger.info("Iniciando identificación...");
        const res = await DefaultService.identifyFingerprintIdentifyPost({
          file,
        });

        logger.info("Identificación completada", {
          matched: res.matched,
          personId: res.person_id,
          score: res.score,
        });

        setResult(res);

        if (res.matched) {
          addToast({
            type: "success",
            title: "Identidad confirmada",
            description: `${res.name} - Confianza: ${((res.score || 0) * 100).toFixed(1)}%`,
            duration: 5000,
          });
        } else {
          addToast({
            type: "info",
            title: "No se encontraron coincidencias",
            description:
              "La huella no coincide con ningún registro en la base de datos",
            duration: 4000,
          });
        }
      } else {
        // Validar campos de registro
        if (!regId.trim() || !regName.trim()) {
          logger.warn("Campos de registro incompletos");
          addToast({
            type: "error",
            title: "Campos incompletos",
            description: "Por favor, completa todos los campos requeridos",
          });
          setLoading(false);
          return;
        }

        logger.info("Registrando huella", {
          personId: regId,
          name: regName,
        });

        const res = await DefaultService.registerFingerprintRegisterPost({
          person_id: regId,
          name: regName,
          document: regId,
          file: file,
        });

        logger.info("Registro completado", {
          success: res.success,
          recordId: res.record_id,
          minutiaeCount: res.minutiae_count,
        });

        if (res.success) {
          addToast({
            type: "success",
            title: "Usuario registrado exitosamente",
            description: res.message,
            duration: 5000,
          });
          setRegName("");
          setRegId("");
          setImagePreview(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = "";
          }
        } else {
          throw new Error("El registro falló sin información adicional");
        }
      }
    } catch (error) {
      logger.error("Error procesando huella", error);

      let errorMessage = "Error procesando la huella";
      let errorDescription =
        "Ocurrió un error inesperado. Por favor, intenta nuevamente.";

      if (error instanceof ApiError) {
        logger.error("Error de API", {
          status: error.status,
          statusText: error.statusText,
          body: error.body,
          url: error.url,
        });

        if (error.status === 0 || error.status >= 500) {
          errorMessage = "Error del servidor";
          errorDescription =
            "El servidor no está respondiendo. Verifica que el backend esté corriendo.";
        } else if (error.status === 400) {
          errorMessage = "Solicitud inválida";
          errorDescription =
            error.body?.detail || "Los datos proporcionados no son válidos.";
        } else if (error.status === 404) {
          errorMessage = "Recurso no encontrado";
          errorDescription = "El endpoint solicitado no existe.";
        } else if (error.status === 503) {
          errorMessage = "Servicio no disponible";
          errorDescription =
            "El servicio está temporalmente no disponible. Intenta más tarde.";
        } else {
          errorDescription =
            error.body?.detail || error.message || errorDescription;
        }
      } else if (error instanceof Error) {
        if (
          error.message.includes("Failed to fetch") ||
          error.message.includes("NetworkError")
        ) {
          errorMessage = "Error de conexión";
          errorDescription =
            "No se pudo conectar con el servidor. Verifica tu conexión y que el backend esté corriendo.";
        } else {
          errorDescription = error.message;
        }
      }

      addToast({
        type: "error",
        title: errorMessage,
        description: errorDescription,
        duration: 6000,
      });
    } finally {
      setLoading(false);
      logger.debug("Procesamiento finalizado");
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-8 font-sans dark">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-border pb-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-primary/10 rounded-full">
              <Fingerprint className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                BioSecure <span className="text-primary">Gov</span>
              </h1>
              <p className="text-muted-foreground">
                Sistema de Identificación Biométrica Gubernamental
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => navigate("/")}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Panel
            </Button>
            <Button
              variant={activeTab === "scan" ? "default" : "outline"}
              onClick={() => setActiveTab("scan")}
            >
              <Search className="w-4 h-4 mr-2" />
              Identificar
            </Button>
            <Button
              variant={activeTab === "register" ? "default" : "outline"}
              onClick={() => setActiveTab("register")}
            >
              <UserPlus className="w-4 h-4 mr-2" />
              Registrar
            </Button>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Column: Scanner */}
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader>
              <CardTitle>Escáner Biométrico</CardTitle>
              <CardDescription>
                Sube una imagen de huella dactilar (BMP, PNG, JPG)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                className="border-2 border-dashed border-border rounded-lg h-64 flex flex-col items-center justify-center cursor-pointer hover:border-primary/50 transition-colors bg-muted/20 overflow-hidden relative"
                onClick={() => fileInputRef.current?.click()}
              >
                {imagePreview ? (
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <>
                    <Upload className="w-12 h-12 text-muted-foreground mb-2" />
                    <p className="text-sm text-muted-foreground">
                      Click para subir imagen
                    </p>
                  </>
                )}
                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </div>

              {activeTab === "register" && (
                <div className="space-y-3 pt-4 animate-in fade-in">
                  <Input
                    placeholder="ID Personal (DNI/Cédula)"
                    value={regId}
                    onChange={(e) => setRegId(e.target.value)}
                  />
                  <Input
                    placeholder="Nombre Completo"
                    value={regName}
                    onChange={(e) => setRegName(e.target.value)}
                  />
                </div>
              )}
            </CardContent>
            <CardFooter>
              <Button
                className="w-full"
                size="lg"
                disabled={
                  !imagePreview ||
                  loading ||
                  (activeTab === "register" && (!regId || !regName))
                }
                onClick={() =>
                  processImage(activeTab === "scan" ? "identify" : "register")
                }
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Procesando...
                  </>
                ) : activeTab === "scan" ? (
                  "Buscar Coincidencia"
                ) : (
                  "Guardar Registro"
                )}
              </Button>
            </CardFooter>
          </Card>

          {/* Right Column: Results */}
          <div className="space-y-6">
            {result && activeTab === "scan" ? (
              <Card
                className={`border-l-4 ${
                  result.matched ? "border-l-green-500" : "border-l-red-500"
                }`}
              >
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {result.matched ? (
                      <>
                        <CheckCircle className="text-green-500" />
                        Identidad Confirmada
                      </>
                    ) : (
                      <>
                        <XCircle className="text-red-500" />
                        No Encontrado
                      </>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {result.matched ? (
                    <div className="space-y-2">
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <span className="text-muted-foreground">Nombre:</span>
                        <span className="font-medium">{result.name}</span>

                        <span className="text-muted-foreground">ID:</span>
                        <span className="font-mono">{result.person_id}</span>

                        <span className="text-muted-foreground">
                          Confianza:
                        </span>
                        <span className="text-green-500 font-bold">
                          {((result.score || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted-foreground">
                      No se encontraron coincidencias en la base de datos con el
                      umbral de confianza actual.
                    </p>
                  )}
                </CardContent>
              </Card>
            ) : (
              <div className="h-full flex items-center justify-center text-muted-foreground opacity-50">
                <div className="text-center">
                  <Search className="w-12 h-12 mx-auto mb-2" />
                  <p>Los resultados aparecerán aquí</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Hidden canvas for minutiae overlay */}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}
