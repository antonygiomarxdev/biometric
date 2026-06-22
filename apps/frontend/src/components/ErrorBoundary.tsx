import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Optional fallback override (e.g. a per-section mini-boundary). */
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Global error boundary.
 *
 * Without this, any uncaught render error in the React tree (bad
 * data, missing field, NaN in a width prop) blanks the page and
 * leaves the user staring at a blank tab.  This catches the error,
 * logs it, and shows a recovery panel with a "Reload" button and a
 * "Reset state" button that wipes the local state and re-renders.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error("[ErrorBoundary] Uncaught render error", error, info);
  }

  private handleReload = (): void => {
    window.location.reload();
  };

  private handleReset = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (this.props.fallback !== undefined) {
      return this.props.fallback;
    }

    return (
      <div
        role="alert"
        className="min-h-screen flex items-center justify-center bg-background text-foreground p-6 font-sans dark"
      >
        <div className="max-w-md w-full bg-card border border-destructive/40 rounded-lg p-6 shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <AlertTriangle className="w-6 h-6 text-destructive shrink-0" />
            <h1 className="text-lg font-bold tracking-tight">
              Algo se rompió
            </h1>
          </div>
          <p className="text-sm text-muted-foreground mb-3">
            La página encontró un error inesperado. Podés recargar o
            intentar limpiar el estado para volver a intentarlo.
          </p>
          <pre className="text-xs font-mono text-destructive/80 bg-destructive/5 border border-destructive/20 rounded p-3 mb-4 overflow-auto max-h-40 whitespace-pre-wrap break-words">
            {error.name}: {error.message}
          </pre>
          <div className="flex items-center gap-2">
            <Button onClick={this.handleReset} variant="outline" size="sm">
              <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
              Reintentar
            </Button>
            <Button onClick={this.handleReload} size="sm">
              Recargar página
            </Button>
          </div>
        </div>
      </div>
    );
  }
}
