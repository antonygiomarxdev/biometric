import {
  Fingerprint,
  Search,
  UserPlus,
  Settings,
  ShieldCheck,
  ScanFace,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { AppMode, BiometricModality } from "@/types/fingerprint";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";

interface SidebarProps {
  activeMode: AppMode;
  onModeChange: (mode: AppMode) => void;
  activeModality: BiometricModality;
  onModalityChange: (modality: BiometricModality) => void;
  className?: string;
}

export function Sidebar({
  activeMode,
  onModeChange,
  activeModality,
  onModalityChange,
  className,
}: SidebarProps) {
  const getModalityLabel = (m: BiometricModality) => {
    switch (m) {
      case "fingerprint":
        return "Dactiloscopia";
      case "face":
        return "Reconocimiento Facial";
      case "iris":
        return "Escáner de Iris";
      default:
        return m;
    }
  };

  const getModalityIcon = (m: BiometricModality) => {
    switch (m) {
      case "fingerprint":
        return <Fingerprint className="h-4 w-4" />;
      case "face":
        return <ScanFace className="h-4 w-4" />;
      default:
        return <Fingerprint className="h-4 w-4" />;
    }
  };

  return (
    <div
      className={cn(
        "pb-12 min-h-screen border-r border-border bg-card",
        className
      )}
    >
      <div className="space-y-4 py-4">
        <div className="px-4 py-2">
          <div className="flex items-center gap-2 mb-6 px-2">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Fingerprint className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-bold tracking-tight text-foreground leading-none">
                BioSecure
              </h2>
              <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
                Unified Identity
              </span>
            </div>
          </div>

          {/* Selector de Modalidad */}
          <div className="mb-6 px-1">
            <label className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2 block px-1">
              Modalidad Biométrica
            </label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  className="w-full justify-between h-9 text-xs"
                >
                  <span className="flex items-center gap-2">
                    {getModalityIcon(activeModality)}
                    {getModalityLabel(activeModality)}
                  </span>
                  <ChevronDown className="h-3 w-3 opacity-50" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-[220px]">
                <DropdownMenuItem
                  onClick={() => onModalityChange("fingerprint")}
                >
                  <Fingerprint className="mr-2 h-4 w-4" />
                  <span>Dactiloscopia</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => onModalityChange("face")}>
                  <ScanFace className="mr-2 h-4 w-4" />
                  <span>Reconocimiento Facial</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <div className="space-y-1">
            <h3 className="mb-2 px-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground/70">
              Operaciones
            </h3>
            <Button
              variant={activeMode === "scan" ? "secondary" : "ghost"}
              className="w-full justify-start gap-3 font-medium"
              onClick={() => onModeChange("scan")}
            >
              <Search className="h-4 w-4" />
              Identificación
            </Button>
            <Button
              variant={activeMode === "register" ? "secondary" : "ghost"}
              className="w-full justify-start gap-3 font-medium"
              onClick={() => onModeChange("register")}
            >
              <UserPlus className="h-4 w-4" />
              Registro Civil
            </Button>
          </div>
        </div>

        <div className="px-4 py-2">
          <h3 className="mb-2 px-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground/70">
            Sistema
          </h3>
          <div className="space-y-1">
            <Button
              variant="ghost"
              className="w-full justify-start gap-3 text-muted-foreground"
            >
              <Settings className="h-4 w-4" />
              Configuración
            </Button>
            <Button
              variant="ghost"
              className="w-full justify-start gap-3 text-muted-foreground"
            >
              <ShieldCheck className="h-4 w-4" />
              Auditoría
            </Button>
          </div>
        </div>
      </div>

      <div className="absolute bottom-4 left-0 w-full px-6">
        <div className="rounded-xl bg-primary/5 p-4 border border-primary/10">
          <div className="flex items-center gap-3">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <p className="text-xs font-medium text-primary">Sistema Activo</p>
          </div>
          <p className="text-[10px] text-muted-foreground mt-1 ml-5">
            v3.0.0 (Multimodal)
          </p>
        </div>
      </div>
    </div>
  );
}
