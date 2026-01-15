import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import type { AppMode, BiometricModality } from "@/types/fingerprint";

interface MainLayoutProps {
  children: ReactNode;
  activeMode: AppMode;
  onModeChange: (mode: AppMode) => void;
  activeModality: BiometricModality;
  onModalityChange: (modality: BiometricModality) => void;
}

export function MainLayout({
  children,
  activeMode,
  onModeChange,
  activeModality,
  onModalityChange,
}: MainLayoutProps) {
  return (
    <div className="flex min-h-screen bg-background text-foreground font-sans">
      <aside className="w-64 fixed inset-y-0 left-0 z-50 hidden md:block">
        <Sidebar
          activeMode={activeMode}
          onModeChange={onModeChange}
          activeModality={activeModality}
          onModalityChange={onModalityChange}
        />
      </aside>
      <main className="flex-1 md:ml-64 min-h-screen">
        <div className="container mx-auto p-8 max-w-7xl animate-in fade-in duration-500">
          {children}
        </div>
      </main>
    </div>
  );
}
